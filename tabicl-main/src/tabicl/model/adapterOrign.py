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
        self._final_proj_in_dim: int | None = None

    def _build_final_proj(self, in_dim: int, device):
        if self.out_dim is None:
            return
        self.final_proj = nn.Linear(in_dim, self.out_dim).to(device)
        self._final_proj_built = True
        self._final_proj_in_dim = int(in_dim)

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
            in_dim = int(C * self.d)
            if (not self._final_proj_built) or (self._final_proj_in_dim != in_dim):
                self._build_final_proj(in_dim, x.device)
            return self.final_proj(z_concat)

        return z_concat

class LoRAResidualAdapter(nn.Module):
    def __init__(self, dim=256, rank=8, lora_alpha=16, fuse="mean", dropout=0.0, gate_eps=1e-3):
        super().__init__()
        self.dim = dim
        self.fuse = fuse
        self.rank = rank
        self.scale = lora_alpha / rank
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.gate_eps = gate_eps

        self.B = nn.Linear(dim, rank, bias=False)
        self.A = nn.Linear(rank, dim, bias=False)

        nn.init.zeros_(self.A.weight)              # delta=0 at init
        nn.init.normal_(self.B.weight, std=0.02)

        # alpha = sigmoid(a) + eps, init a=-6 => sigmoid ~ 0.0025
        self.a = nn.Parameter(torch.tensor(-6.0))

        self.gamma = nn.Parameter(torch.zeros(dim))
        self.beta  = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        base = x[:, 0, :] if self.fuse == "first" else x.mean(dim=1)
        base = base * (1 + self.gamma) + self.beta
        h = self.drop(base)

        delta = self.A(self.B(h)) * self.scale
        alpha = torch.sigmoid(self.a) + self.gate_eps   # 非零 => A 有梯度
        return base + alpha * delta



class SafeResidualAdapter(nn.Module):
    """
    UCR-safe adapter:
      - Input:  (N, C, D)
      - Output: (N, D)   (matches TabICL input dim when D=256)
    Key property:
      - Initialized to (almost) exact identity for C=1
      - Learns a residual delta with a learnable gate alpha (init=0)
    """
    def __init__(self, dim=256, hidden=256, dropout=0.0, fuse="mean", out_dim: int | None = None):
        super().__init__()
        assert fuse in ["mean", "first", "concat"], "fuse must be 'mean', 'first', or 'concat'"
        self.dim = dim
        self.fuse = fuse
        self.out_dim = out_dim

        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Optional final projection (useful for fuse='concat' to keep TabICL input dim fixed)
        self.final_proj = nn.Identity()
        self._final_proj_built = False
        self._final_proj_in_dim: int | None = None

    def _build_final_proj(self, in_dim: int, device):
        if self.out_dim is None:
            return
        if self.out_dim == in_dim:
            self.final_proj = nn.Identity()
        else:
            self.final_proj = nn.Linear(in_dim, self.out_dim).to(device)
        self._final_proj_built = True
        self._final_proj_in_dim = int(in_dim)

        # Learnable residual scale; init=0 => exact baseline (for C=1)
        #self.alpha = nn.Parameter(torch.tensor(0.0001))

        # **critical**: make residual branch start at zero
        # nn.init.zeros_(self.fc2.weight)
        # nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, D)
        if x.dim() != 3:
            raise ValueError(f"Input must be (N, C, D). Got {tuple(x.shape)}")
        N, C, D = x.shape
        if D != self.dim:
            raise ValueError(f"Expected dim={self.dim}, but got D={D}")

        if self.fuse in ["mean", "first"]:
            # Baseline embedding:
            # - UCR: C=1 => mean == first == x[:,0,:]
            if self.fuse == "first":
                base = x[:, 0, :]
            else:
                base = x.mean(dim=1)

            # Residual delta
            h = F.gelu(self.fc1(base))
            h = self.drop(h)
            delta = self.fc2(h)
            out = base + delta

            if self.out_dim is not None:
                in_dim = int(out.shape[-1])
                if (not self._final_proj_built) or (self._final_proj_in_dim != in_dim):
                    self._build_final_proj(in_dim, x.device)
                return self.final_proj(out)
            return out

        # fuse == 'concat': process each channel independently, then concat
        x_flat = x.reshape(N * C, D)  # (N*C, D)
        h = F.gelu(self.fc1(x_flat))
        h = self.drop(h)
        delta = self.fc2(h)  # (N*C, D)
        z = (x_flat + delta).reshape(N, C, D)  # (N, C, D)
        z_concat = z.reshape(N, C * D)  # (N, C*D)

        if self.out_dim is not None:
            in_dim = int(C * D)
            if (not self._final_proj_built) or (self._final_proj_in_dim != in_dim):
                self._build_final_proj(in_dim, x.device)
            return self.final_proj(z_concat)
        return z_concat
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


class CALDA_AdapterV2(nn.Module):
    """
    CALDA v2: Per-channel pathway + concat.

    For multivariate time-series input x with shape (B, C, mantis_emb_dim):
      - For each channel independently, run:
          MCM (with C=1) -> bottleneck1 -> bottleneck2
        producing z_c of shape (B, tabicl_input_dim)
      - Concatenate over channels to get final output:
          z_out of shape (B, C * tabicl_input_dim)

    Notes:
      - We reuse the same weights across channels.
      - This keeps channels separated (no cross-channel mixing) while still
        using the same CALDA blocks.
    """

    def __init__(self, mantis_emb_dim: int = 256, tabicl_input_dim: int = 128, out_dim: int | None = None):
        super().__init__()
        self.mantis_emb_dim = mantis_emb_dim
        self.tabicl_input_dim = tabicl_input_dim
        self.out_dim = out_dim

        # Same modules as CALDA_Adapter, but applied per-channel.
        self.mcm = MultivariateChannelMixer(emb_dim=mantis_emb_dim)
        self.bottleneck = nn.Linear(mantis_emb_dim, tabicl_input_dim)
        # self.bottleneck2 = nn.Linear(512, tabicl_input_dim)

        # Optional final projection: (C * tabicl_input_dim) -> out_dim
        # Built lazily on first forward because C can vary across datasets.
        self.final_proj = nn.Identity()
        self._final_proj_built = False
        self._final_proj_in_dim: int | None = None

    def _build_final_proj(self, in_dim: int, device):
        if self.out_dim is None:
            return
        self.final_proj = nn.Linear(in_dim, self.out_dim).to(device)
        self._final_proj_built = True
        self._final_proj_in_dim = int(in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, mantis_emb_dim)
        returns: (B, C * tabicl_input_dim)
        """
        if x.dim() != 3:
            raise ValueError(
                f"Input must be (Batch, Channels, Emb_Dim). Got shape {tuple(x.shape)}"
            )

        B, C, D = x.shape
        if D != self.mantis_emb_dim:
            raise ValueError(f"Expected mantis_emb_dim={self.mantis_emb_dim}, but got D={D}")

        # Process each channel independently by folding C into batch.
        # (B, C, D) -> (B*C, 1, D)
        x_per_channel = x.reshape(B * C, 1, D)

        # 1) MCM (trivial self-attn for channel length=1) + pooling => (B*C, D)
        z_mixed = self.mcm(x_per_channel)

        # 2) Bottleneck => (B*C, tabicl_input_dim)
        z_proj = self.bottleneck(z_mixed)

        # (B*C, tabicl_input_dim) -> (B, C, tabicl_input_dim) -> (B, C*tabicl_input_dim)
        z_proj = z_proj.reshape(B, C, self.tabicl_input_dim)
        z_concat = z_proj.reshape(B, C * self.tabicl_input_dim)

        if self.out_dim is not None:
            in_dim = int(C * self.tabicl_input_dim)
            if (not self._final_proj_built) or (self._final_proj_in_dim != in_dim):
                self._build_final_proj(in_dim, x.device)
            return self.final_proj(z_concat)

        return z_concat 
     
    
class CALDA_Adapter(nn.Module):
    """
    Full CALDA Architecture: Channel-Aware Latent Distribution Alignment
    Connecting Mantis (Time Series) -> TabICL (Tabular).
    """
    def __init__(self, mantis_emb_dim=256, tabicl_input_dim=128):
        super().__init__()
        
        # 1. MCM: Handle topological loss & multi-variate dependency
        self.mcm = MultivariateChannelMixer(emb_dim=mantis_emb_dim)
        self.bottleneck1 = nn.Linear(mantis_emb_dim, 512)
        # 2. Bottleneck Projection: Reduce dimension (256 -> 128)
        self.bottleneck2 = nn.Linear(512, tabicl_input_dim)
        
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
        z_proj = self.bottleneck1(z_mixed) # -> (Batch, TabICL_Input_Dim)
        z_proj = self.bottleneck2(z_proj)
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

class CausalChannelAdapter(nn.Module):
    """
    Causal Channel Adapter:
    Treats Channels as Tokens and uses Attention to fuse them into a unified representation.
    
    Flow:
    1. Input: (B, C, Mantis_Dim)
    2. Positional Encoding for Channels (Optional but recommended if channel order matters)
    3. Self-Attention over Channels: Allows channels to "talk" to each other (discover causality).
    4. Cross-Attention / Pooling: Compresses C channels into fixed K tokens for TabICL.
    """
    def __init__(self, mantis_emb_dim=256, tabicl_input_dim=256, num_latents=4, dropout=0.1):
        super().__init__()
        self.mantis_emb_dim = mantis_emb_dim
        self.tabicl_input_dim = tabicl_input_dim
        
        # 1. Channel Mixer (Self-Attention over C)
        # We project Mantis Dim to an internal working dimension
        self.input_proj = nn.Linear(mantis_emb_dim, tabicl_input_dim)
        
        self.channel_encoder_layer = nn.TransformerEncoderLayer(
            d_model=tabicl_input_dim, 
            nhead=4, 
            dim_feedforward=tabicl_input_dim*2, 
            dropout=dropout, 
            batch_first=True
        )
        self.channel_encoder = nn.TransformerEncoder(self.channel_encoder_layer, num_layers=2)
        
        # 2. Latent Query (The "Causal Summary" Token)
        # We learn a fixed number of latent queries to summarize the multivariate state
        # Usually 1 token is enough to represent the "System State" for classification
        self.num_latents = num_latents
        self.latent_query = nn.Parameter(torch.randn(1, num_latents, tabicl_input_dim))
        
        # 3. Cross Attention: Latent (Q) attends to Channels (K, V)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=tabicl_input_dim, 
            num_heads=4, 
            dropout=dropout, 
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(tabicl_input_dim)
        self.out_proj = nn.Linear(tabicl_input_dim, tabicl_input_dim)

    def forward(self, x):
        """
        x: (B, C, mantis_emb_dim)
        """
        B, C, D = x.shape
        
        # Project to working dimension
        # (B, C, D) -> (B, C, tabicl_dim)
        h = self.input_proj(x)
        
        # --- Stage 1: Channel Discovery (Self-Attention) ---
        # "Which channels are related?"
        # Input to TransformerEncoder is (B, Seq_Len=C, Dim)
        h_mixed = self.channel_encoder(h) 
        
        # --- Stage 2: Causal Fusion (Cross-Attention) ---
        # "Summarize the system state into fixed vector(s)"
        # Q: Latent Query expanded to batch (B, num_latents, dim)
        q = self.latent_query.repeat(B, 1, 1)
        
        # K, V: The mixed channel representations
        k = v = h_mixed
        
        # attn_output: (B, num_latents, dim)
        attn_output, _ = self.cross_attn(query=q, key=k, value=v)
        
        # Residual connection + Norm on the latent query
        z = self.norm(q + attn_output)
        
        # Final projection
        z = self.out_proj(z)
        
        # Flatten: (B, num_latents * dim)
        # If num_latents=1, this ensures output is exactly (B, tabicl_input_dim)
        return z.reshape(B, -1)




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