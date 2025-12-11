from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, List

import torch
from torch import nn, Tensor
import numpy as np

from .tabicl import TabICL
from .inference_config import InferenceConfig
from tabicl.model.mantis_dev.architecture.architecture import Mantis8M
from tabicl.model.mantis_dev.trainer.trainer import MantisTrainer

DEFAULT_MANTIS_PRETRAINED = Path(
    "/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/Models/Mantis/Mantis_cheickpoint/"
)

def _load_generic_state_dict(model: nn.Module, checkpoint: dict, strict: bool = True) -> None:
    """Load a checkpoint that may store the state dict under different keys."""
    state_dict_keys: List[str] = [
        "state_dict",
        "model_state_dict",
        "model",
        "network",
        "checkpoint",
    ]

    state_dict = None
    for key in state_dict_keys:
        if key in checkpoint and isinstance(checkpoint[key], dict):
            state_dict = checkpoint[key]
            break

    if state_dict is None:
        if all(isinstance(k, str) for k in checkpoint.keys()):
            state_dict = checkpoint
        else:
            raise ValueError("Unable to locate state dict within the provided checkpoint")

    cleaned_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=strict)

    if missing_keys:
        warnings.warn(f"Missing keys when loading state dict: {missing_keys}")
    if unexpected_keys:
        warnings.warn(f"Unexpected keys when loading state dict: {unexpected_keys}")


def build_mantis_encoder(
    mantis_checkpoint: str | Path | None,
    device: torch.device,
    hidden_dim: int = 256,
    seq_len: int = 512,
) -> Mantis8M:
    """Instantiate Mantis8M and load weights from a checkpoint or pretrained dir."""

    mantis_model = Mantis8M(
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_patches=32,
        scalar_scales=None,
        hidden_dim_scalar_enc=32,
        epsilon_scalar_enc=1.1,
        transf_depth=6,
        transf_num_heads=8,
        transf_mlp_dim=512,
        transf_dim_head=128,
        transf_dropout=0.1,
        device=str(device),
        pre_training=False,
    )

    checkpoint_path = Path(mantis_checkpoint) if mantis_checkpoint else None
    if checkpoint_path and checkpoint_path.is_dir():
        mantis_model = mantis_model.from_pretrained(str(checkpoint_path))
        print(f"[MantisTabICL] Loaded pretrained Mantis encoder from {checkpoint_path}")
    elif checkpoint_path and checkpoint_path.is_file():
        state = torch.load(checkpoint_path, map_location="cpu")
        _load_generic_state_dict(mantis_model, state, strict=False)
    elif DEFAULT_MANTIS_PRETRAINED.is_dir():
        mantis_model = mantis_model.from_pretrained(str(DEFAULT_MANTIS_PRETRAINED))
        print(f"[MantisTabICL] Loaded default pretrained Mantis encoder from {DEFAULT_MANTIS_PRETRAINED}")
    else:
        raise FileNotFoundError(
            "No valid Mantis checkpoint provided and default directory not found."
        )

    mantis_model.to(device)
    # mantis_model.eval()
    return mantis_model


@torch.no_grad()
def encode_with_mantis(
    mantis_model: Mantis8M,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 16,
) -> np.ndarray:
    """Encode a matrix (n_samples, seq_len) using the provided Mantis encoder."""

    if X.ndim != 3:
        raise ValueError(f"Mantis encoder expects a 3D array, got shape {X.shape}")

    tensor = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
    # tensor = tensor.unsqueeze(1)
    model = MantisTrainer(device=device, network=mantis_model)
    # outputs = []
    # for start in range(0, tensor.size(0), batch_size):
    #     end = start + batch_size
    #     batch = tensor[start:end]
    #     reps =  model.transform(batch,to_numpy=True) # mantis_model(batch)
    #     # outputs.append(reps.cpu())
    #     outputs.append(reps)
    print("mantis 多通道时序 各个通道单独提取")
    z  =   model.transform(tensor,batch_size,to_numpy=True)
    
    return z


# @torch.no_grad()
# def encode_with_mantis(
#     mantis_model: Mantis8M,
#     X: np.ndarray,
#     device: torch.device,
#     batch_size: int = 16,
# ) -> np.ndarray:
#     """Encode a matrix (n_samples, seq_len) using the provided Mantis encoder."""

#     if X.ndim not in (2, 3):
#         raise ValueError(f"Mantis encoder expects a 2D or 3D array, got shape {X.shape}")

#     tensor = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
#     if tensor.ndim == 2:
#         tensor = tensor.unsqueeze(1)

#     embeddings = []
#     for start in range(0, tensor.size(0), batch_size):
#         end = start + batch_size
#         batch = tensor[start:end]
#         reps = mantis_model(batch)
#         embeddings.append(reps.detach().cpu())

#     return torch.cat(embeddings, dim=0).numpy()


class MantisTabICL(nn.Module):
    """Composite model that feeds Mantis embeddings into a TabICL backbone."""

    def __init__(
        self,
        tabicl_checkpoint: str | Path,
        mantis_checkpoint: str | Path,
        mantis_batch_size: int = 32,
        device: Optional[str | torch.device] = None,
    ) -> None:
        super().__init__()

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.mantis_batch_size = mantis_batch_size

        tabicl_checkpoint = Path(tabicl_checkpoint)
        mantis_checkpoint = Path(mantis_checkpoint)

        tabicl_state = torch.load(tabicl_checkpoint, map_location="cpu")
        assert "config" in tabicl_state and "state_dict" in tabicl_state, (
            "Provided TabICL checkpoint must contain both 'config' and 'state_dict' entries."
        )

        self.tabicl_model = TabICL(**tabicl_state["config"])
        self.tabicl_model.load_state_dict(tabicl_state["state_dict"])
        self.tabicl_model.to(self.device)
        self.max_classes = self.tabicl_model.max_classes
        self.tabicl_config = tabicl_state["config"]

        self.mantis_model = build_mantis_encoder(
            mantis_checkpoint=mantis_checkpoint,
            device=self.device,
            hidden_dim=256,
            seq_len=512,
        )

        # self.tabicl_model.eval()
        # self.mantis_model.eval()

    def _current_device(self) -> torch.device:
        return next(self.tabicl_model.parameters()).device


    # def _encode_with_mantis(self, X: Tensor, device: Optional[torch.device] = None) -> Tensor:
    #     """Run the Mantis encoder over flattened batches to obtain per-row embeddings."""
        
    #     if X.ndim != 3:
    #         raise ValueError(f"Mantis encoder expects a 3D array, got shape {X.shape}")
        
    #     B, T, H = X.shape
    #     device = device or next(self.mantis_model.parameters()).device
        
    #     # Ensure input is a tensor on the correct device
    #     if not isinstance(X, torch.Tensor):
    #         X = torch.from_numpy(np.asarray(X, dtype=np.float32))
    #     X = X.to(device)

    #     # Reshape to (B*T, 1, H) for Mantis processing
    #     # Mantis expects (Batch, SeqLen, Channels) where SeqLen=1 for row-wise encoding
    #     X_reshaped = X.reshape(-1, 1, H)
        
    #     # Process in batches to avoid OOM if necessary, but keep gradients
    #     # If mantis_batch_size is large enough, we can process all at once
    #     # For training, we usually rely on the outer loop (Trainer) to handle batching via micro-batches
    #     # So we can process the whole micro-batch here.
        
    #     # However, if B*T is very large, we might still want to chunk it.
    #     # But chunking with gradients requires gathering the results.
        
    #     representations_list = []
    #     total_samples = X_reshaped.size(0)
        
    #     for start in range(0, total_samples, self.mantis_batch_size):
    #         end = min(start + self.mantis_batch_size, total_samples)
    #         batch = X_reshaped[start:end]
    #         # Forward pass through Mantis
    #         reps = self.mantis_model(batch)
    #         representations_list.append(reps)
            
    #     # Concatenate results
    #     representations = torch.cat(representations_list, dim=0)
        
    #     # Reshape back to (B, T, D)
    #     return representations.reshape(B, T, -1)
        
    #     X_flat = tensor.reshape(-1, 1, H)
    #     outputs = []
    #     for i in range(0, X_flat.size(0), self.mantis_batch_size):
    #         batch = X_flat[i : i + self.mantis_batch_size]
    #         out = self.mantis_model(batch)
    #         outputs.append(out)
            
    #     concatenated = torch.cat(outputs, dim=0)
    #     return concatenated.reshape(B, T, -1)
    
    
    def _encode_with_mantis(self, X: Tensor, device: Optional[torch.device] = None) -> Tensor:
        """Run the Mantis encoder over flattened batches to obtain per-row embeddings."""
        
        if X.ndim != 3:
            raise ValueError(f"Mantis encoder expects a 3D array, got shape {X.shape}")
        
        B, T, H = X.shape
        # device = device or next(self.mantis_model.parameters()).device
        
        # Ensure input is a tensor on the correct device
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        X = X.to(self.device)

        # Reshape to (B*T, 1, H) for Mantis processing
        # Mantis expects (Batch, SeqLen, Channels) where SeqLen=1 for row-wise encoding
        X_reshaped = X.reshape(-1, 1, H)
        
        # Forward pass through Mantis
        # We process all at once since micro-batching is handled externally in Trainer
        reps = self.mantis_model(X_reshaped)
        
        # Reshape back to (B, T, D)
        return reps.reshape(B, T, -1)    
    def forward(
        self,
        X: Tensor,
        y_train: Tensor,
        feature_shuffles: Optional[List[List[int]]] = None,
        embed_with_test: bool = False,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        inference_config: Optional[InferenceConfig] = None,
    ) -> Tensor:
        device = self._current_device()
        X = X.to(device).float()
        y_train = y_train.to(device)

        mantis_device = next(self.mantis_model.parameters()).device
        mantis_repr = self._encode_with_mantis(X, device=mantis_device)
        mantis_repr = mantis_repr.float()

        return self.tabicl_model(
            mantis_repr,
            y_train,
            feature_shuffles=feature_shuffles,
            embed_with_test=embed_with_test,
            return_logits=return_logits,
            softmax_temperature=softmax_temperature,
            inference_config=inference_config,
        )
