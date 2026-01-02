import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelMLPConcatAdapter(nn.Module):
    """
    Adapter: per-channel 2-layer MLP -> d-dim, then concat across channels.

    Input:  x of shape (N, C, D)
    Output: z_concat of shape (N, C*d)

    Optional: if out_dim is set, will project (C*d) -> out_dim.
    """
    def __init__(
        self,
        in_dim: int = 256,     # D
        hidden_dim: int = 256, # hidden size for MLP
        out_per_channel: int = 64,  # d
        dropout: float = 0.0,
        out_dim: int | None = None, # optional final projection, e.g. TabICL input dim
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.d = out_per_channel
        self.out_dim = out_dim

        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, out_per_channel))

        self.channel_mlp = nn.Sequential(*layers)
        self.ln = nn.LayerNorm(out_per_channel) if use_layernorm else nn.Identity()

        # Final projection (optional): (C*d) -> out_dim
        # Use a valid placeholder module; real Linear is built lazily on first forward if out_dim is set.
        self.final_proj = nn.Identity()
        self._final_proj_built = False

    def _build_final_proj(self, C: int, device):
        if self.out_dim is None:
            return
        self.final_proj = nn.Linear(C * self.d, self.out_dim).to(device)
        self._final_proj_built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, C, D)
        returns:
          - if out_dim is None: (N, C*d)
          - else:              (N, out_dim)
        """
        if x.dim() != 3:
            raise ValueError(f"Input must be (N, C, D). Got shape {tuple(x.shape)}")

        N, C, D = x.shape
        if D != self.in_dim:
            raise ValueError(f"Expected in_dim={self.in_dim}, but got D={D}")

        # Apply per-channel MLP: reshape to (N*C, D) -> (N*C, d) -> (N, C, d)
        z = x.reshape(N * C, D)
        z = self.channel_mlp(z)              # (N*C, d)
        z = self.ln(z)                       # (N*C, d)
        z = z.reshape(N, C, self.d)          # (N, C, d)

        # Concat across channels: (N, C*d)
        z_concat = z.reshape(N, C * self.d)

        # Optional final projection to match TabICL input dim
        if self.out_dim is not None:
            if not self._final_proj_built:
                self._build_final_proj(C, x.device)
            return self.final_proj(z_concat)

        return z_concat

class SafeResidualAdapter(nn.Module):
    """
    UCR-safe adapter:
      - Input:  (N, C, D)
      - Output: (N, D)   (matches TabICL input dim when D=256)
    Key property:
      - Initialized to (almost) exact identity for C=1
      - Learns a residual delta with a learnable gate alpha (init=0)
    """
    def __init__(self, dim=256, hidden=256, dropout=0.0, fuse="mean"):
        super().__init__()
        assert fuse in ["mean", "first"], "fuse must be 'mean' or 'first'"
        self.dim = dim
        self.fuse = fuse

        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Learnable residual scale; init=0 => exact baseline (for C=1)
        self.alpha = nn.Parameter(torch.tensor(0.0))

        # **critical**: make residual branch start at zero
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, D)
        if x.dim() != 3:
            raise ValueError(f"Input must be (N, C, D). Got {tuple(x.shape)}")
        N, C, D = x.shape
        if D != self.dim:
            raise ValueError(f"Expected dim={self.dim}, but got D={D}")

        # Baseline embedding:
        # - UCR: C=1 => mean == first == x[:,0,:]
        if self.fuse == "first":
            base = x[:, 0, :]
        else:
            base = x.mean(dim=1)

        # Residual delta
        h = F.gelu(self.fc1(base))
        h = self.drop(h)
        delta = self.fc2(h)  # starts at 0 exactly

        return base + self.alpha * delta
class GatedAttentionPooling(nn.Module):
    """
    Gated Attention Pooling Layer for MCM.
    Aggregates channel dimension based on learnable importance weights.
    Formula: z_mixed = sum(softmax(W*Z + b) * Z)
    """
    def __init__(self, input_dim):
        super().__init__()
        self.gate_layer = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        Input: (Batch, Channels, Emb_Dim)
        Output: (Batch, Emb_Dim)
        """
        # Calculate attention scores: (Batch, Channels, 1)
        attn_logits = self.gate_layer(x)
        # Softmax over channels dimension (dim=1)
        attn_weights = F.softmax(attn_logits, dim=1)
        
        # Weighted sum: (Batch, Channels, Emb_Dim) * (Batch, Channels, 1) -> sum over dim 1
        # Result: (Batch, Emb_Dim)
        x_pooled = (x * attn_weights).sum(dim=1)
        return x_pooled

class MultivariateChannelMixer(nn.Module):
    """
    Module 1: Multivariate Channel Mixer (MCM)
    Models dependencies between channels using Self-Attention and aggregates them.
    """
    def __init__(self, emb_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=emb_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.pooling = GatedAttentionPooling(emb_dim)

    def forward(self, x):
        """
        Input: (Batch, Channels, Emb_Dim)
        Output: (Batch, Emb_Dim)
        """
        # Self-Attention over Channels dimension
        # Query=Key=Value=x
        attn_out, _ = self.self_attn(query=x, key=x, value=x)
        
        # Residual connection + LayerNorm
        x = self.norm(x + attn_out)
        
        # Gated Attention Pooling to collapse Channel dimension
        x_mixed = self.pooling(x)
        return x_mixed

class HeterogeneousDistributionProjector(nn.Module):
    """
    Module 3: Heterogeneous Distribution Projector (HDP) - Optimized
    Splits features into groups and applies different activation functions.
    Includes stability improvements for power operations.
    """
    def __init__(self, input_dim=128, num_groups=4):
        super().__init__()
        assert input_dim % num_groups == 0, "Input dim must be divisible by num_groups"
        self.group_dim = input_dim // num_groups
        self.num_groups = num_groups

    def forward(self, x):
        """
        Input: (Batch, Input_Dim)
        Output: (Batch, Input_Dim) with diverse distributions
        """
        # Split into groups: List of (Batch, Group_Dim)
        groups = torch.chunk(x, self.num_groups, dim=1)
        
        processed_groups = []
        
        # Group 1: Identity (Simulates Normal/Gaussian distribution)
        g1 = groups[0]
        processed_groups.append(g1)
        
        # Group 2: Tanh (Simulates Bounded distribution [-1, 1])
        g2 = torch.tanh(groups[1])
        processed_groups.append(g2)
        
        # Group 3: Signed Power x^3 (Simulates Long-tail distribution)
        # Improvement: Added clamping to prevent gradient explosion
        g3_in = groups[2]
        # Clamping absolute value to 5.0 prevents values > 125.0
        # This keeps gradients manageable while preserving the shape
        g3_safe = torch.clamp(g3_in, min=-5.0, max=5.0) 
        g3 = torch.sign(g3_safe) * torch.pow(torch.abs(g3_safe), 3)
        processed_groups.append(g3)
        
        # Group 4: ReLU (Simulates Sparse/Counting distribution)
        g4 = F.relu(groups[3])
        processed_groups.append(g4)
        
        # Concatenate back
        z_out = torch.cat(processed_groups, dim=1)
        return z_out

class CALDA_Adapter(nn.Module):
    """
    Full CALDA Architecture: Channel-Aware Latent Distribution Alignment
    Connecting Mantis (Time Series) -> TabICL (Tabular).
    """
    def __init__(self, mantis_emb_dim=256, tabicl_input_dim=128):
        super().__init__()
        
        # 1. MCM: Handle topological loss & multi-variate dependency
        self.mcm = MultivariateChannelMixer(emb_dim=mantis_emb_dim)
        
        # 2. Bottleneck Projection: Reduce dimension (256 -> 128)
        self.bottleneck = nn.Linear(mantis_emb_dim, tabicl_input_dim)
        
        # 3. HDP: Handle distributional misalignment (White -> Diverse)
        #self.hdp = HeterogeneousDistributionProjector(input_dim=tabicl_input_dim)

    def forward(self, x):
        """
        Input: (Batch, Channels, Mantis_Emb_Dim)
        Output: (Batch, TabICL_Input_Dim)
        """
        # Handle flattened input check
        if x.dim() != 3:
             raise ValueError(f"Input must be (Batch, Channels, Emb_Dim). Got shape {x.shape}")
            
        # 1. MCM
        z_mixed = self.mcm(x) # -> (Batch, Mantis_Emb_Dim)
        
        # 2. Bottleneck
        z_proj = self.bottleneck(z_mixed) # -> (Batch, TabICL_Input_Dim)
        
        # 3. HDP
        #z_out = self.hdp(z_proj) # -> (Batch, TabICL_Input_Dim)
        
        return z_proj

class DistributionDiversityLoss(nn.Module):
    """
    Auxiliary Loss for Stage 1 Training (Adapter Warmup) - Optimized.
    Maximizes the variance of Skewness and Kurtosis across feature dimensions.
    """
    def __init__(self, eps=1e-5): # Increased eps for stability
        super().__init__()
        self.eps = eps

    def calc_moments(self, x):
        """
        Calculate Skewness and Kurtosis for each feature across the batch.
        x: (Batch, Features)
        Returns: skew (Features,), kurt (Features,)
        """
        # Add small noise to prevent zero variance division if features collapse
        if self.training:
            x = x + torch.randn_like(x) * 1e-6

        mean = torch.mean(x, dim=0)
        std = torch.std(x, dim=0) + self.eps
        
        # Center the data
        centered = x - mean
        
        # Normalize
        norm_x = centered / std
        
        # Skewness = E[((x-mu)/sigma)^3]
        skew = torch.mean(torch.pow(norm_x, 3), dim=0)
        
        # Kurtosis = E[((x-mu)/sigma)^4]
        kurt = torch.mean(torch.pow(norm_x, 4), dim=0)
        
        return skew, kurt

    def forward(self, z_out):
        """
        z_out: Output from CALDA Adapter (Batch, Features)
        """
        # Ensure batch size is sufficient for statistical validity
        if z_out.size(0) < 4:
            return torch.tensor(0.0, device=z_out.device, requires_grad=True)

        skew, kurt = self.calc_moments(z_out)
        
        # Calculate variance of these moments across the feature dimension
        var_skew = torch.var(skew)
        var_kurt = torch.var(kurt)
        
        # Loss: minimize negative variance (maximize diversity)
        loss_div = - (var_skew + var_kurt)
        return loss_div

if __name__ == "__main__":
    # Test Block
    BATCH_SIZE = 64 # Increased batch size for better moment estimation
    CHANNELS = 30
    MANTIS_DIM = 256
    TABICL_DIM = 128
    
    adapter = CALDA_Adapter(mantis_emb_dim=MANTIS_DIM, tabicl_input_dim=TABICL_DIM)
    div_loss_fn = DistributionDiversityLoss()
    
    # Dummy Input
    x = torch.randn(BATCH_SIZE, CHANNELS, MANTIS_DIM)
    
    # Forward
    z = adapter(x)
    print(f"Output shape: {z.shape}")
    
    # Loss
    loss = div_loss_fn(z)
    print(f"Diversity Loss: {loss.item()}")
    
    # Gradient Check (Simple)
    loss.backward()
    print("Backward pass successful.")