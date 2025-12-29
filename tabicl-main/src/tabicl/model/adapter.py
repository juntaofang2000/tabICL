import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Literal, Optional, Tuple


class CausalDisentanglerAdapter(nn.Module):
    """Cross-attention based adapter for multivariate time-series channel embeddings.

    Design goals:
    - No pooling/summing over the source channel dimension.
    - Learn a small set of latent queries that attend to source channels, producing
      a fixed number of latent "feature columns".

    Input:
        x: (B, Source_Channels, Emb_Dim)
    Output:
        z: (B, Num_Latents, Emb_Dim)
    """

    def __init__(
        self,
        emb_dim: int,
        num_latents: int = 10,
        num_heads: int = 4,
        dropout: float = 0.0,
        norm: str = "bn",
        use_affine_norm: bool = False,
    ):
        super().__init__()

        if emb_dim % num_heads != 0:
            raise ValueError(f"emb_dim ({emb_dim}) must be divisible by num_heads ({num_heads}).")

        self.emb_dim = int(emb_dim)
        self.num_latents = int(num_latents)
        self.num_heads = int(num_heads)

        self.latent_queries = nn.Parameter(torch.randn(1, self.num_latents, self.emb_dim) * 0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.emb_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Output normalization: stabilize distribution for downstream TabICL augmentations.
        # BatchNorm1d expects (N, C, L). We treat Emb_Dim as channels.
        norm = (norm or "").lower()
        if norm in {"bn", "batchnorm", "batchnorm1d"}:
            self.out_norm = nn.BatchNorm1d(self.emb_dim, affine=use_affine_norm)
            self._norm_kind = "bn"
        elif norm in {"ln", "layernorm"}:
            self.out_norm = nn.LayerNorm(self.emb_dim, elementwise_affine=use_affine_norm)
            self._norm_kind = "ln"
        elif norm in {"none", "", None}:
            self.out_norm = nn.Identity()
            self._norm_kind = "none"
        else:
            raise ValueError(f"Unknown norm='{norm}'. Use 'bn', 'ln', or 'none'.")

        # A small residual MLP helps expressiveness without collapsing channels.
        self.ffn = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim * 4),
            nn.GELU(),
            nn.Linear(self.emb_dim * 4, self.emb_dim),
        )
        self.pre_norm = nn.LayerNorm(self.emb_dim)
        self.post_norm = nn.LayerNorm(self.emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Input must be (Batch, Source_Channels, Emb_Dim). Got shape {tuple(x.shape)}")
        if x.size(-1) != self.emb_dim:
            raise ValueError(
                f"Last dim mismatch: expected Emb_Dim={self.emb_dim}, got {x.size(-1)}."
            )

        B = x.size(0)
        q = self.latent_queries.expand(B, -1, -1)  # (B, Num_Latents, Emb_Dim)

        # Cross-attention: Query=latent queries, Key/Value=source channels.
        # No pooling over channels; attention learns to extract causal latents.
        attn_out, _ = self.cross_attn(query=q, key=x, value=x, need_weights=False)

        # Residual + FFN (Transformer-style)
        z = self.pre_norm(attn_out)
        z = z + self.ffn(z)
        z = self.post_norm(z)

        # Distribution alignment normalization
        if self._norm_kind == "bn":
            # (B, Num_Latents, Emb_Dim) -> (B, Emb_Dim, Num_Latents)
            z = self.out_norm(z.transpose(1, 2)).transpose(1, 2)
        else:
            z = self.out_norm(z)

        return z


class CausalGNNLayer(nn.Module):
    """A simple graph message-passing layer over K latent nodes.

    Given latents Z with shape (B, K, D) and an adjacency matrix A with shape (K, K),
    performs:
        Z_new = Z + ReLU( A @ Z @ W )

    where (A @ Z)[b, i, :] = sum_j A[i, j] * Z[b, j, :]
    """

    def __init__(self, emb_dim: int, bias: bool = True):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.proj = nn.Linear(self.emb_dim, self.emb_dim, bias=bias)

    def forward(self, z: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        if z.dim() != 3:
            raise ValueError(f"z must be (B, K, D). Got shape {tuple(z.shape)}")
        if A.dim() != 2 or A.size(0) != A.size(1):
            raise ValueError(f"A must be (K, K). Got shape {tuple(A.shape)}")
        if z.size(1) != A.size(0):
            raise ValueError(
                f"K mismatch: z has K={z.size(1)} but A is {tuple(A.shape)}"
            )

        # (B, K, D)
        msg = torch.einsum("ij,bjd->bid", A, z)
        msg = self.proj(msg)
        return z + F.relu(msg)


class StructuralCausalAdapter(nn.Module):
    """Structural causal adapter with (i) disentanglement and (ii) structure learning.

    Pipeline:
      1) Disentanglement Phase:
         - Cross-attention extracts K latent variables Z in R^{B x K x D}.
         - Independence regularization encourages distinct causal mechanisms.
      2) Causal Discovery Phase:
         - Learn a global, sparse adjacency A in {0,1}^{KxK} (relaxed via Gumbel-Sigmoid).
         - Graph refinement updates latents using a causal GNN layer.

    Forward returns:
        output_features: (B, K, D)
        aux_loss_dict: {"independence_loss": ..., "sparsity_loss": ..., "adjacency": ..., "adjacency_prob": ...}
    """

    def __init__(
        self,
        emb_dim: int,
        num_latents: int = 10,
        num_heads: int = 4,
        dropout: float = 0.0,
        norm: str = "bn",
        use_affine_norm: bool = False,
        independence: Literal["orth", "hsic"] = "orth",
        hsic_kernel: Literal["rbf", "linear"] = "rbf",
        hsic_sigma: float = 1.0,
        gumbel_tau: float = 1.0,
        gumbel_hard: bool = False,
        allow_self_edges: bool = False,
        sparsity_on: Literal["prob", "sample"] = "prob",
    ):
        super().__init__()

        if emb_dim % num_heads != 0:
            raise ValueError(f"emb_dim ({emb_dim}) must be divisible by num_heads ({num_heads}).")

        self.emb_dim = int(emb_dim)
        self.num_latents = int(num_latents)
        self.num_heads = int(num_heads)
        self.independence = independence
        self.hsic_kernel = hsic_kernel
        self.hsic_sigma = float(hsic_sigma)
        self.gumbel_tau = float(gumbel_tau)
        self.gumbel_hard = bool(gumbel_hard)
        self.allow_self_edges = bool(allow_self_edges)
        self.sparsity_on = sparsity_on

        # ---- 1) Disentanglement (cross-attn latent extraction) ----
        self.latent_queries = nn.Parameter(torch.randn(1, self.num_latents, self.emb_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.emb_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True,
        )

        norm_lc = (norm or "").lower()
        if norm_lc in {"bn", "batchnorm", "batchnorm1d"}:
            self.out_norm = nn.BatchNorm1d(self.emb_dim, affine=use_affine_norm)
            self._norm_kind = "bn"
        elif norm_lc in {"ln", "layernorm"}:
            self.out_norm = nn.LayerNorm(self.emb_dim, elementwise_affine=use_affine_norm)
            self._norm_kind = "ln"
        elif norm_lc in {"none", "", None}:
            self.out_norm = nn.Identity()
            self._norm_kind = "none"
        else:
            raise ValueError(f"Unknown norm='{norm}'. Use 'bn', 'ln', or 'none'.")

        self.ffn = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim * 4),
            nn.GELU(),
            nn.Linear(self.emb_dim * 4, self.emb_dim),
        )
        self.pre_norm = nn.LayerNorm(self.emb_dim)
        self.post_norm = nn.LayerNorm(self.emb_dim)

        # ---- 2) Causal discovery (global structure learning + graph refinement) ----
        # Learn global adjacency logits (shared across batch).
        # We'll sample a relaxed adjacency via Gumbel-Sigmoid for differentiability.
        self.adj_logits = nn.Parameter(torch.zeros(self.num_latents, self.num_latents))
        self.gnn = CausalGNNLayer(emb_dim=self.emb_dim)

        if not self.allow_self_edges:
            self.register_buffer("_diag_mask", (1.0 - torch.eye(self.num_latents)), persistent=False)
        else:
            self.register_buffer("_diag_mask", torch.ones(self.num_latents, self.num_latents), persistent=False)

    @staticmethod
    def _gumbel_sigmoid(
        logits: torch.Tensor,
        tau: float = 1.0,
        hard: bool = False,
        eps: float = 1e-10,
    ) -> torch.Tensor:
        """Sample from the Binary Concrete / Gumbel-Sigmoid distribution."""
        if tau <= 0:
            raise ValueError(f"tau must be > 0, got {tau}")
        u = torch.rand_like(logits)
        g = -torch.log(-torch.log(u.clamp_min(eps)).clamp_min(eps))
        y = torch.sigmoid((logits + g) / tau)
        if hard:
            y_hard = (y > 0.5).to(y.dtype)
            y = (y_hard - y).detach() + y
        return y

    @staticmethod
    def _orthogonality_loss(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Encourage pairwise orthogonality among the K latents (per sample).

        z: (B, K, D)
        Returns scalar loss.
        """
        z_norm = z / (z.norm(dim=-1, keepdim=True) + eps)
        # Gram: (B, K, K)
        gram = torch.matmul(z_norm, z_norm.transpose(1, 2))
        off_diag = gram - torch.diag_embed(torch.diagonal(gram, dim1=1, dim2=2))
        return (off_diag ** 2).mean()

    @staticmethod
    def _center_gram(K: torch.Tensor) -> torch.Tensor:
        n = K.size(0)
        eye = torch.eye(n, device=K.device, dtype=K.dtype)
        ones = torch.ones(n, n, device=K.device, dtype=K.dtype) / float(n)
        H = eye - ones
        return H @ K @ H

    def _hsic_pair(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Empirical HSIC(x, y) using either linear or RBF kernel.

        x, y: (N, D)
        Returns scalar HSIC.
        """
        if x.dim() != 2 or y.dim() != 2 or x.size(0) != y.size(0):
            raise ValueError(f"x,y must be (N,D) with same N. Got {tuple(x.shape)}, {tuple(y.shape)}")
        N = x.size(0)
        if N < 2:
            return x.new_tensor(0.0)

        if self.hsic_kernel == "linear":
            Kx = x @ x.t()
            Ky = y @ y.t()
        else:
            # RBF
            x2 = (x ** 2).sum(dim=1, keepdim=True)
            y2 = (y ** 2).sum(dim=1, keepdim=True)
            dx = x2 + x2.t() - 2.0 * (x @ x.t())
            dy = y2 + y2.t() - 2.0 * (y @ y.t())
            sigma2 = (self.hsic_sigma ** 2)
            Kx = torch.exp(-dx / (2.0 * sigma2))
            Ky = torch.exp(-dy / (2.0 * sigma2))

        Kx = self._center_gram(Kx)
        Ky = self._center_gram(Ky)
        hsic = (Kx * Ky).sum() / ((N - 1) ** 2)
        return hsic

    def compute_independence_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Compute independence loss among K latents.

        Supported:
          - "orth": orthogonality loss on latent vectors.
          - "hsic": average pairwise HSIC across latents (computed on flattened embeddings).
        """
        if z.dim() != 3:
            raise ValueError(f"Expected z to be (B, K, D). Got shape {tuple(z.shape)}")

        if self.independence == "orth":
            return self._orthogonality_loss(z)

        # HSIC mode: treat each latent k as a variable with samples across batch.
        # We flatten embedding dimensions into a vector per sample.
        B, K, D = z.shape
        if B < 4 or K < 2:
            return z.new_tensor(0.0)
        z_flat = z.reshape(B, K, D)
        total = 0.0
        count = 0
        for i in range(K):
            xi = z_flat[:, i, :]
            for j in range(i + 1, K):
                yj = z_flat[:, j, :]
                total = total + self._hsic_pair(xi, yj)
                count += 1
        return total / max(count, 1)

    def sample_adjacency(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (A_sample, A_prob) each with shape (K, K)."""
        logits = self.adj_logits * self._diag_mask
        A_prob = torch.sigmoid(logits)
        A_sample = self._gumbel_sigmoid(logits, tau=self.gumbel_tau, hard=self.gumbel_hard)
        A_sample = A_sample * self._diag_mask
        return A_sample, A_prob

    def compute_sparsity_loss(self, A_sample: torch.Tensor, A_prob: torch.Tensor) -> torch.Tensor:
        if self.sparsity_on == "sample":
            return A_sample.abs().mean()
        return A_prob.abs().mean()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if x.dim() != 3:
            raise ValueError(f"Input must be (Batch, Source_Channels, Emb_Dim). Got shape {tuple(x.shape)}")
        if x.size(-1) != self.emb_dim:
            raise ValueError(f"Last dim mismatch: expected Emb_Dim={self.emb_dim}, got {x.size(-1)}.")

        B = x.size(0)
        q = self.latent_queries.expand(B, -1, -1)  # (B, K, D)

        # Cross-attention: Query=latent queries, Key/Value=source channels.
        z_attn, _ = self.cross_attn(query=q, key=x, value=x, need_weights=False)

        # Residual + FFN (Transformer-style)
        z = self.pre_norm(z_attn)
        z = z + self.ffn(z)
        z = self.post_norm(z)

        # Distribution alignment normalization
        if self._norm_kind == "bn":
            z = self.out_norm(z.transpose(1, 2)).transpose(1, 2)
        else:
            z = self.out_norm(z)

        # ---- Independence of mechanisms ----
        independence_loss = self.compute_independence_loss(z)

        # ---- Global causal structure learning + graph refinement ----
        # A_sample, A_prob = self.sample_adjacency()
        # sparsity_loss = self.compute_sparsity_loss(A_sample, A_prob)
        # z_refined = self.gnn(z, A_sample)

        # aux: Dict[str, torch.Tensor] = {
        #     "independence_loss": independence_loss,
        #     "sparsity_loss": sparsity_loss,
        #     # Provide adjacency tensors for logging/analysis.
        #     "adjacency": A_sample,
        #     "adjacency_prob": A_prob,
        # }
        #return z_refined, aux
        aux: Dict[str, torch.Tensor] = {
             "independence_loss": independence_loss,}
        return  z,aux
        
        
        


class ICLAlignmentLoss(nn.Module):
    """Loss that aligns adapter outputs to an ICL-friendly metric space.

    Components:
    1) Prototypical Networks loss computed from a support/query split.
    2) KL regularizer that encourages features to match N(0, I) (approximate) to
       reduce distribution shift under TabICL-style quantile/power transforms.

    Expected usage:
        z = adapter(x)  # (B, Num_Latents, Emb_Dim)
        loss = loss_fn(z, y, n_support=K)
    """

    def __init__(
        self,
        n_support: int = 5,
        kl_weight: float = 1e-2,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.n_support = int(n_support)
        self.kl_weight = float(kl_weight)
        self.eps = float(eps)

    @staticmethod
    def _flatten_features(z: torch.Tensor) -> torch.Tensor:
        # Accept (B, D) or (B, L, D). Flatten latents into feature dimension.
        if z.dim() == 2:
            return z
        if z.dim() == 3:
            return z.reshape(z.size(0), -1)
        raise ValueError(f"Expected z to have 2 or 3 dims, got shape {tuple(z.shape)}")

    def _proto_logits_from_episode(
        self,
        z_support: torch.Tensor,
        y_support: torch.Tensor,
        z_query: torch.Tensor,
        y_query: torch.Tensor,
    ):
        """Compute ProtoNet logits for a single episode.

        Inputs:
            z_support: (N_support, D) or (N_support, L, D)
            y_support: (N_support,)
            z_query: (N_query, D) or (N_query, L, D)
            y_query: (N_query,)
        Returns:
            logits: (N_query_valid, N_classes)
            y_query_mapped: (N_query_valid,)
        """
        z_s = self._flatten_features(z_support)
        z_q = self._flatten_features(z_query)

        y_s = y_support.view(-1)
        y_q = y_query.view(-1)

        device = z_s.device
        classes = torch.unique(y_s)
        classes = classes[torch.argsort(classes)]
        if classes.numel() < 2:
            return None, None

        # Prototypes from support
        prototypes = []
        for c in classes:
            idx = torch.nonzero(y_s == c, as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            prototypes.append(z_s[idx].mean(dim=0))
        if len(prototypes) < 2:
            return None, None
        prototypes = torch.stack(prototypes, dim=0)  # (N_classes, D)

        # Map query labels into [0..N_classes-1], drop unknown labels
        max_label = int(max(int(y_s.max().item()), int(y_q.max().item())))
        mapper = torch.full((max_label + 1,), -1, dtype=torch.long, device=device)
        mapper[classes] = torch.arange(classes.numel(), device=device)
        y_q_mapped = mapper[y_q]
        valid = y_q_mapped != -1
        if not valid.any():
            return None, None

        z_q = z_q[valid]
        y_q_mapped = y_q_mapped[valid]

        # Squared euclidean distances -> logits = -dist
        z_q2 = (z_q ** 2).sum(dim=1, keepdim=True)
        p2 = (prototypes ** 2).sum(dim=1).unsqueeze(0)
        cross = z_q @ prototypes.t()
        dists = z_q2 + p2 - 2.0 * cross
        logits = -dists
        return logits, y_q_mapped

    def _proto_logits(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        n_support: int,
    ):
        """Compute ProtoNet logits for all query samples in the batch.

        Returns:
            logits: (N_query, N_classes)
            y_query_mapped: (N_query,)
        """
        if y.dim() != 1:
            y = y.view(-1)

        device = z.device
        classes = torch.unique(y)
        classes = classes[torch.argsort(classes)]
        n_classes = classes.numel()

        # Build support/query indices per class.
        support_indices = []
        query_indices = []
        query_class_targets = []

        for class_idx, c in enumerate(classes):
            idx = torch.nonzero(y == c, as_tuple=False).view(-1)
            if idx.numel() < n_support + 1:
                # Need at least 1 query sample to contribute.
                continue
            sup = idx[:n_support]
            qry = idx[n_support:]
            support_indices.append(sup)
            query_indices.append(qry)
            query_class_targets.append(torch.full((qry.numel(),), class_idx, device=device, dtype=torch.long))

        if len(support_indices) < 2:
            # Not enough classes with query samples to form meaningful episodic loss.
            return None, None

        # Prototypes: mean of support embeddings per class.
        prototypes = []
        kept_classes = []
        for class_idx, sup in enumerate(support_indices):
            prototypes.append(z[sup].mean(dim=0))
            kept_classes.append(class_idx)
        prototypes = torch.stack(prototypes, dim=0)  # (N_kept_classes, D)

        # Queries
        qry_idx = torch.cat(query_indices, dim=0)
        z_q = z[qry_idx]  # (N_query, D)
        y_q = torch.cat(query_class_targets, dim=0)  # (N_query,)

        # Squared euclidean distances -> logits = -dist
        # dist(i, j) = ||z_q[i] - proto[j]||^2
        z_q2 = (z_q ** 2).sum(dim=1, keepdim=True)  # (N_query, 1)
        p2 = (prototypes ** 2).sum(dim=1).unsqueeze(0)  # (1, N_kept)
        cross = z_q @ prototypes.t()  # (N_query, N_kept)
        dists = z_q2 + p2 - 2.0 * cross
        logits = -dists

        return logits, y_q

    def _kl_to_standard_normal(self, z_flat: torch.Tensor) -> torch.Tensor:
        """Approximate KL between N(mu, diag(var)) and N(0, I).

        We estimate per-dimension mean/var over the current batch and penalize:
            KL = 0.5 * sum(mu^2 + var - log(var) - 1)
        """
        mu = z_flat.mean(dim=0)
        var = z_flat.var(dim=0, unbiased=False).clamp_min(self.eps)
        kl_per_dim = 0.5 * (mu ** 2 + var - torch.log(var) - 1.0)
        return kl_per_dim.mean()

    def forward(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        n_support: int | None = None,
    ) -> torch.Tensor:
        n_support = self.n_support if n_support is None else int(n_support)
        if n_support < 1:
            raise ValueError(f"n_support must be >= 1, got {n_support}")

        z_flat = self._flatten_features(z)
        logits_y = self._proto_logits(z_flat, y, n_support=n_support)

        if logits_y[0] is None:
            # Fall back: only KL regularization if episodic split isn't viable.
            return self.kl_weight * self._kl_to_standard_normal(z_flat)

        logits, y_query = logits_y
        proto_loss = F.cross_entropy(logits, y_query)
        kl_loss = self._kl_to_standard_normal(z_flat)
        return proto_loss + self.kl_weight * kl_loss

    def forward_episode(
        self,
        z_support: torch.Tensor,
        y_support: torch.Tensor,
        z_query: torch.Tensor,
        y_query: torch.Tensor,
    ) -> torch.Tensor:
        """Explicit episodic ProtoNet + KL.

        This matches the typical ICL setting where you already have a support/query split.
        """
        logits_y = self._proto_logits_from_episode(z_support, y_support, z_query, y_query)
        if logits_y[0] is None:
            z_all = torch.cat([
                self._flatten_features(z_support),
                self._flatten_features(z_query),
            ], dim=0)
            return self.kl_weight * self._kl_to_standard_normal(z_all)

        logits, y_q = logits_y
        proto_loss = F.cross_entropy(logits, y_q)
        z_all = torch.cat([
            self._flatten_features(z_support),
            self._flatten_features(z_query),
        ], dim=0)
        #kl_loss = self._kl_to_standard_normal(z_all)
        #return proto_loss + self.kl_weight * kl_loss
        return proto_loss

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