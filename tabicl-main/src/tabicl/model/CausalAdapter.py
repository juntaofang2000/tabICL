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


class RobustCWA(nn.Module):
    """
    Robust Causal Whitening Adapter (RCWA) - Optimized Version
    ----------------------------------------------------------
    针对小 Batch Size (UCR) 优化的因果适配器。
    
    关键改进:
    1. "Compress-Then-Whiten": 先降维到 64 维，再进行白化。解决 N < D 导致的秩亏问题。
    2. Residual Connection: 增加线性旁路，保证初始性能不低于 Linear Projection。
    3. Stabilized Whitening: 针对低维特征的稳健 Newton-Schulz。
    """
    
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=64, dropout=0.1, 
                 iter_num=5):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.iter_num = iter_num
        
        # --- 模块 1: 压缩与非线性解混 (Compression & Unmixing) ---
        # 768 -> 256 -> 64
        # 这一步将高维纠缠信号映射到低维潜在因果空间
        self.pre_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim) # 输出 64 维
        )
        
        # 残差旁路: 一个简单的线性映射，保证梯度畅通
        self.residual_proj = nn.Linear(input_dim, output_dim)
        
        # --- 模块 2: 正交旋转 (在低维空间进行) ---
        # 64 -> 64
        self.ortho_fc = nn.Linear(output_dim, output_dim, bias=False)
        
        # --- 模块 4: 稀疏门控 ---
        self.gate_val = nn.Linear(output_dim, output_dim)
        self.gate_mask = nn.Linear(output_dim, output_dim)
        
        # 初始化
        nn.init.orthogonal_(self.ortho_fc.weight)
        nn.init.constant_(self.gate_mask.bias, 3.0) # 初始全开，避免早期阻断
        
        # 动态调整门控的缩放，防止输出过小
        self.output_scale = nn.Parameter(torch.ones(1) * 2.0)

    def _newton_schulz_whitening(self, X):
        """
        稳健的低维 Newton-Schulz 白化
        X shape: [Batch, 64] -> 可行！
        """
        B, D = X.shape
        orig_dtype = X.dtype
        
        # 0. 均值中心化
        mu = X.mean(dim=0, keepdim=True)
        X = X - mu
        
        X_double = X.double()
        
        # 1. 计算协方差 (Gram Matrix)
        # 此时 D=64, B=32，矩阵仍然可能是奇异的，需要强正则化
        sigma = torch.mm(X_double.t(), X_double) / (B - 1 if B > 1 else 1)
        
        # 强正则化: 保证矩阵可逆
        sigma += 1e-3 * torch.eye(D, device=X.device, dtype=torch.double)
        
        # 2. 归一化 (使用 Trace)
        trace_sigma = torch.trace(sigma)
        sigma_norm = trace_sigma * 1.5 + 1e-6
        sigma_scaled = sigma / sigma_norm
        
        # 3. Newton-Schulz 迭代
        W = torch.eye(D, device=X.device, dtype=torch.double)
        
        for _ in range(self.iter_num):
            P = torch.mm(torch.mm(W, sigma_scaled), W.t())
            term = 1.5 * torch.eye(D, device=X.device, dtype=torch.double) - 0.5 * P
            W = torch.mm(term, W)
            
        # 4. 应用白化
        X_normalized = X_double / torch.sqrt(sigma_norm)
        X_white_double = torch.mm(X_normalized, W.t())
        
        return X_white_double.to(dtype=orig_dtype), W.to(dtype=orig_dtype)

    def forward(self, z_mantis, training_mode=False):
        # 0. 输入清洗
        if torch.isnan(z_mantis).any() or torch.isinf(z_mantis).any():
            z_mantis = torch.nan_to_num(z_mantis, nan=0.0)
            z_mantis = torch.clamp(z_mantis, -10.0, 10.0)

        # 1. 降维解混
        h_latent = self.pre_mlp(z_mantis)
        
        # 残差连接
        h_res = self.residual_proj(z_mantis)
        h_latent = h_latent + h_res 

        # 2. 正交旋转
        h_orth = self.ortho_fc(h_latent)
        
        # 3. 统计白化
        # 如果 Batch Size 太小 (< Dim)，白化是不稳定的。
        # 对于 64 维特征，建议 Batch Size >= 64。如果是 32，我们会自动降级为 LayerNorm。
        if h_orth.size(0) > self.output_dim: 
            h_white, W_white = self._newton_schulz_whitening(h_orth)
        else:
            # Fallback: 小 Batch 使用 InstanceNorm/LayerNorm 模拟白化
            # 这种情况下，我们放弃"去相关"，只做"标准化"
            h_white = F.layer_norm(h_orth, h_orth.shape[1:])
            W_white = torch.eye(h_orth.shape[1], device=h_orth.device)
            
        # 4. 稀疏门控
        # 增加 output_scale 确保进入 TabICL 的数值范围合适
        val = self.gate_val(h_white)
        mask = torch.sigmoid(self.gate_mask(h_white))
        x_tab = val * mask * self.output_scale
        
        if training_mode:
            return x_tab, h_orth, self.ortho_fc.weight, W_white
        
        return x_tab

class RobustCausalLoss(nn.Module):
    def __init__(self, lambda_orth=0.1, lambda_indep=0.05):
        super().__init__()
        self.lambda_orth = lambda_orth
        self.lambda_indep = lambda_indep
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, predictions, targets, h_orth, W_orth, x_tab):
        # 1. 任务损失
        task_loss = self.ce_loss(predictions, targets)
        
        # 2. 正交性约束 (针对 64x64 矩阵，非常快且稳定)
        I = torch.eye(W_orth.size(1), device=W_orth.device)
        orth_loss = torch.norm(W_orth.t() @ W_orth - I, p='fro')
        
        # 3. 独立性损失
        B, D = x_tab.size()
        if B > 4: # 至少几个样本才能算协方差
            mu = x_tab.mean(dim=0, keepdim=True)
            centered = x_tab - mu
            cov = (centered.t() @ centered) / (B - 1)
            # 惩罚非对角元素
            indep_loss = torch.norm(cov - torch.diag(torch.diag(cov)), p='fro')
        else:
            indep_loss = torch.tensor(0.0, device=x_tab.device)
            
        total_loss = task_loss + self.lambda_orth * orth_loss + self.lambda_indep * indep_loss
        
        if torch.isnan(total_loss):
            return torch.tensor(0.0, requires_grad=True, device=x_tab.device), {}
            
        return total_loss, {"task": task_loss, "orth": orth_loss, "indep": indep_loss}