import os
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ColumnRecoloringConfig:
    mode: str = "none"  # {none, affine}
    scale_min: float = 0.5
    scale_max: float = 2.0
    bias_std: float = 0.0
    seed: int = 42


class ColumnRecoloring(nn.Module):
    """Column-wise recoloring for tabular features.

    Applies per-column affine transform:
        X_new[..., j] = a[j] * X[..., j] + b[j]

    Notes:
    - a,b are registered buffers (no gradients) and follow device.
    - a,b are initialized once (train-side) and can be saved/loaded.
    """

    def __init__(self, *, mode: str = "none"):
        super().__init__()
        self.mode = str(mode)

        self.register_buffer("a", torch.empty(0), persistent=True)
        self.register_buffer("b", torch.empty(0), persistent=True)
        self._printed_stats = False

    @property
    def enabled(self) -> bool:
        return self.mode == "affine"

    @property
    def dim(self) -> int | None:
        if self.a.numel() == 0:
            return None
        return int(self.a.numel())

    def is_initialized(self) -> bool:
        return self.enabled and (self.a.numel() > 0) and (self.b.numel() > 0)

    @staticmethod
    def _build_a(D: int, *, scale_min: float, scale_max: float, device: torch.device) -> torch.Tensor:
        if D <= 0:
            raise ValueError(f"D must be > 0, got {D}")
        smin = float(scale_min)
        smax = float(scale_max)
        if smin <= 0 or smax <= 0:
            raise ValueError(f"scale_min/scale_max must be > 0, got {smin}, {smax}")
        if smax < smin:
            raise ValueError(f"scale_max must be >= scale_min, got {smin}, {smax}")

        log_min = torch.log(torch.tensor(smin, device=device, dtype=torch.float32))
        log_max = torch.log(torch.tensor(smax, device=device, dtype=torch.float32))
        return torch.exp(torch.linspace(log_min, log_max, steps=int(D), device=device, dtype=torch.float32))

    @staticmethod
    def _build_b(D: int, *, bias_std: float, seed: int, device: torch.device) -> torch.Tensor:
        std = float(bias_std)
        if std <= 0:
            return torch.zeros(int(D), device=device, dtype=torch.float32)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        b_cpu = torch.randn(int(D), generator=gen, dtype=torch.float32) * std
        return b_cpu.to(device)

    def initialize(
        self,
        *,
        D: int,
        scale_min: float,
        scale_max: float,
        bias_std: float,
        seed: int,
        device: torch.device,
    ) -> None:
        if not self.enabled:
            return

        a = self._build_a(int(D), scale_min=scale_min, scale_max=scale_max, device=device)
        b = self._build_b(int(D), bias_std=bias_std, seed=seed, device=device)

        # overwrite buffers in-place to keep registration
        self.a = a
        self.b = b
        self._printed_stats = False

        self._sanity_check_buffers()

    def _sanity_check_buffers(self) -> None:
        if not self.enabled:
            return
        if self.a.numel() == 0 or self.b.numel() == 0:
            raise RuntimeError("ColumnRecoloring is enabled but buffers are empty.")
        if self.a.shape != self.b.shape:
            raise RuntimeError(f"a/b shape mismatch: {tuple(self.a.shape)} vs {tuple(self.b.shape)}")
        if not torch.isfinite(self.a).all():
            raise RuntimeError("ColumnRecoloring buffer a contains NaN/inf")
        if not torch.isfinite(self.b).all():
            raise RuntimeError("ColumnRecoloring buffer b contains NaN/inf")

    def cache_path(
        self,
        *,
        cache_dir: str,
        D: int,
        scale_min: float,
        scale_max: float,
        bias_std: float,
        seed: int,
        prefix: str = "recoloring",
    ) -> str:
        os.makedirs(cache_dir, exist_ok=True)
        name = (
            f"{prefix}_{self.mode}_D{int(D)}"
            f"_smin{float(scale_min):g}_smax{float(scale_max):g}"
            f"_bstd{float(bias_std):g}_seed{int(seed)}.pt"
        )
        return os.path.join(cache_dir, name)

    def load_or_initialize(
        self,
        *,
        cache_path: str,
        D: int,
        scale_min: float,
        scale_max: float,
        bias_std: float,
        seed: int,
        device: torch.device,
    ) -> str:
        """Load recoloring buffers from cache_path, or initialize+save if missing.

        Caller should ensure this happens using train split only (no test leakage).

        Returns: "loaded" or "initialized".
        """
        if not self.enabled:
            return "disabled"

        if cache_path and os.path.isfile(cache_path):
            payload = torch.load(cache_path, map_location="cpu")
            a = payload.get("a", None)
            b = payload.get("b", None)
            if a is None or b is None:
                raise RuntimeError(f"Invalid recoloring cache (missing a/b): {cache_path}")
            a = torch.as_tensor(a, dtype=torch.float32, device=device)
            b = torch.as_tensor(b, dtype=torch.float32, device=device)
            if a.numel() != int(D) or b.numel() != int(D):
                raise RuntimeError(
                    f"Recoloring cache dim mismatch: cache D={int(a.numel())}, expected D={int(D)} ({cache_path})"
                )
            self.a = a
            self.b = b
            self._printed_stats = False
            self._sanity_check_buffers()
            return "loaded"

        self.initialize(
            D=D,
            scale_min=scale_min,
            scale_max=scale_max,
            bias_std=bias_std,
            seed=seed,
            device=device,
        )

        if cache_path:
            try:
                torch.save(
                    {
                        "a": self.a.detach().cpu(),
                        "b": self.b.detach().cpu(),
                        "config": {
                            "mode": self.mode,
                            "D": int(D),
                            "scale_min": float(scale_min),
                            "scale_max": float(scale_max),
                            "bias_std": float(bias_std),
                            "seed": int(seed),
                        },
                    },
                    cache_path,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to save recoloring cache to {cache_path}: {e}")

        return "initialized"

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return X

        if not self.is_initialized():
            raise RuntimeError("ColumnRecoloring is enabled but not initialized/loaded.")

        if X.numel() == 0:
            return X

        D = X.shape[-1]
        if self.a.numel() != int(D):
            raise RuntimeError(f"ColumnRecoloring D mismatch: X has D={int(D)} but a has D={int(self.a.numel())}")

        shape = [1] * (X.dim() - 1) + [int(D)]
        a = self.a.view(*shape)
        b = self.b.view(*shape)

        Y = X * a + b

        # unit-test/assert style checks
        if Y.shape != X.shape:
            raise RuntimeError(f"Recoloring changed shape: {tuple(X.shape)} -> {tuple(Y.shape)}")
        if not torch.isfinite(Y).all():
            raise RuntimeError("Recoloring produced NaN/inf")

        if not self._printed_stats:
            with torch.no_grad():
                a_min = float(self.a.min().item())
                a_max = float(self.a.max().item())
                b_mean = float(self.b.mean().item())
                b_std = float(self.b.std(unbiased=False).item())
            print(f"[Recoloring] a range: [{a_min:.6g}, {a_max:.6g}] | b mean/std: {b_mean:.6g}/{b_std:.6g}")
            self._printed_stats = True

        return Y
