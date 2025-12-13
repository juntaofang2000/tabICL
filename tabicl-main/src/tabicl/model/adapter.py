import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.hdp = HeterogeneousDistributionProjector(input_dim=tabicl_input_dim)

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
        z_out = self.hdp(z_proj) # -> (Batch, TabICL_Input_Dim)
        
        return z_out

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