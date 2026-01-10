from __future__ import annotations

import os
import sys
import timeit
import warnings
import functools
from contextlib import nullcontext
from pathlib import Path

import math
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.multiprocessing import set_start_method
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from tqdm import tqdm
import wandb

# Ensure we import the local workspace package (repo_root/src/tabicl)
_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from tabicl import TabICL
from tabicl.prior.dataset import PriorDataset
from tabicl.prior.genload import LoadPriorDataset
from tabicl.train.optim import get_scheduler
from tabicl.train.train_config import build_parser

from tabicl.model.mantis_tabicl import build_mantis_encoder
from tabicl.model.mantis_adapter_icl import MantisAdapterICL, TokenMLPAdapter


warnings.filterwarnings(
    "ignore", message=".*The PyTorch API of nested tensors is in prototype stage.*", category=UserWarning
)


class Timer:
    def __enter__(self):
        self.start_time = timeit.default_timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = timeit.default_timer() - self.start_time
        return False


def ddp_cleanup(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        finally:
            if self.ddp:
                destroy_process_group()

    return wrapper


def _load_tabicl_checkpoint(path: str, device: torch.device) -> tuple[TabICL, dict]:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")

    if "config" not in ckpt:
        raise ValueError("TabICL checkpoint must contain 'config'.")

    state_dict = ckpt.get("state_dict")
    if state_dict is None:
        # fallback to some common keys
        for k in ("model_state_dict", "model"):
            if k in ckpt and isinstance(ckpt[k], dict):
                state_dict = ckpt[k]
                break
    if state_dict is None or not isinstance(state_dict, dict):
        raise ValueError("TabICL checkpoint must contain a model state dict ('state_dict' or similar).")

    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model = TabICL(**ckpt["config"])
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[Warn] TabICL missing keys (strict=False): {len(missing)}")
    if unexpected:
        print(f"[Warn] TabICL unexpected keys (strict=False): {len(unexpected)}")

    model.to(device)
    return model, ckpt["config"]


class Trainer:
    def __init__(self, config):
        self.config = config
        if self.config.checkpoint_dir is None:
            self.config.checkpoint_dir = "checkpoints"
        self.configure_ddp()
        self.configure_wandb()

        if self.master_process:
            print("=" * 40)
            print("Training Configuration:")
            for key, value in sorted(vars(self.config).items()):
                print(f"{key}: {value}")
            print("=" * 40)

        self.build_model()
        self.configure_prior()
        self.configure_optimizer()
        self.configure_amp()
        self.load_checkpoint()

    def configure_ddp(self):
        self.ddp = int(os.environ.get("RANK", -1)) != -1

        if self.ddp:
            init_process_group(backend="nccl")
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            self.master_process = self.ddp_rank == 0
            self.config.device = f"cuda:{self.ddp_local_rank}"
            torch.cuda.set_device(self.config.device)

            original_batch_size = self.config.batch_size
            self.config.batch_size = math.ceil(original_batch_size / self.ddp_world_size)

            if self.master_process:
                print(f"DDP training with {self.ddp_world_size} processes")
                if original_batch_size % self.ddp_world_size == 0:
                    print(f"Per-GPU batch size: {self.config.batch_size}")
                else:
                    print(
                        f"Original batch size ({original_batch_size}) cannot be divided by world size ({self.ddp_world_size}).\n"
                        f"Use ceiling division for equal per-GPU batch size: {self.config.batch_size}.\n"
                        f"Effective batch size is {self.config.batch_size * self.ddp_world_size}.\n"
                    )
        else:
            self.master_process = True
            self.ddp_rank = 0
            self.ddp_world_size = 1
            self.ddp_local_rank = 0
            print("No DDP training")

        self.curr_step = 0

        seed_offset = self.ddp_rank if self.ddp else 0
        np.random.seed(self.config.np_seed + seed_offset)
        torch.manual_seed(self.config.torch_seed + seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def configure_wandb(self):
        if self.config.wandb_log and self.master_process:
            id_path = os.path.join(self.config.checkpoint_dir, "wand_id.txt")
            if self.config.wandb_id is None:
                if os.path.exists(id_path):
                    with open(id_path, "r") as f:
                        self.config.wandb_id = f.read().strip()

            self.wandb_run = wandb.init(
                dir=self.config.wandb_dir,
                project=self.config.wandb_project,
                name=self.config.wandb_name,
                id=self.config.wandb_id,
                config=self.config,
                resume="allow",
                mode=self.config.wandb_mode,
            )

            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            with open(id_path, "w") as f:
                f.write(self.wandb_run.id)
        else:
            self.wandb_run = None

    def build_model(self):
        device = torch.device(self.config.device)

        # 1) Load TabICL checkpoint and extract ICL predictor
        tabicl_ckpt = str(self.config.tabicl_ckpt)
        tabicl_model, tabicl_config = _load_tabicl_checkpoint(tabicl_ckpt, device=device)
        icl_predictor = tabicl_model.icl_predictor

        # Freeze ICL parameters
        for p in icl_predictor.parameters():
            p.requires_grad_(False)
        icl_predictor.eval()

        # 2) Build mantis encoder from given checkpoint
        mantis_ckpt = str(self.config.mantis_ckpt)
        mantis_model = build_mantis_encoder(
            mantis_checkpoint=Path(mantis_ckpt),
            device=device,
            hidden_dim=int(self.config.mantis_hidden_dim),
            seq_len=int(self.config.mantis_seq_len),
        )
        for p in mantis_model.parameters():
            p.requires_grad_(False)
        mantis_model.eval()

        # 3) Adapter: mantis_dim -> icl_dim
        # icl_dim is implied by TabICL architecture
        icl_dim = int(tabicl_config["embed_dim"]) * int(tabicl_config["row_num_cls"])
        adapter_hidden_dim = None if self.config.adapter_hidden_dim is None else int(self.config.adapter_hidden_dim)
        adapter = TokenMLPAdapter(
            mantis_dim=int(getattr(mantis_model, "hidden_dim", self.config.mantis_hidden_dim)),
            icl_dim=icl_dim,
            hidden_dim=adapter_hidden_dim,
            dropout=float(self.config.adapter_dropout),
            use_layernorm=not bool(self.config.adapter_no_layernorm),
        ).to(device)

        model = MantisAdapterICL(
            mantis_model=mantis_model,
            icl_predictor=icl_predictor,
            adapter=adapter,
            mantis_seq_len=int(self.config.mantis_seq_len),
            mantis_batch_size=int(self.config.mantis_batch_size),
        ).to(device)
        model.freeze_mantis_and_icl()

        if self.master_process:
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model params: total={total}, trainable={trainable} (adapter-only)")

        if self.config.model_compile:
            model = torch.compile(model, dynamic=True)
            if self.master_process:
                print("Model compiled successfully.")

        if self.ddp:
            self.model = DDP(model, device_ids=[self.ddp_local_rank], broadcast_buffers=False)
            self.raw_model = self.model.module
        else:
            self.model = model
            self.raw_model = model

    def configure_prior(self):
        if self.config.prior_type == "mixup":
            try:
                from tabicl.prior.mixup_dataset import MixupPriorDataset, load_mixup_config
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "prior_type='mixup' requested, but tabicl.prior.mixup_dataset is missing. "
                    "Either add that module to the repo or use a different --prior_type (e.g. mix_scm/real)."
                ) from e

            mixup_config = {
                "n_bit": self.config.mixup_n_bit,
                "n_step": self.config.mixup_n_step,
                "max_class": self.config.mixup_max_class,
                "mix_alpha": self.config.mixup_alpha,
                "augment_cap": self.config.mixup_augment_cap,
            }
            mixup_config.update(load_mixup_config(self.config.mixup_config))
            dataset = MixupPriorDataset(
                batch_size=self.config.batch_size,
                mixup_config=mixup_config,
                min_seq_len=self.config.min_seq_len,
                max_seq_len=self.config.max_seq_len,
                min_train_size=self.config.min_train_size,
                max_train_size=self.config.max_train_size,
            )
        elif self.config.prior_dir is None:
            dataset = PriorDataset(
                real_data_dir=self.config.real_data_dir,
                batch_size=self.config.batch_size,
                batch_size_per_gp=self.config.batch_size_per_gp,
                min_features=self.config.min_features,
                max_features=self.config.max_features,
                max_classes=self.config.max_classes,
                min_seq_len=self.config.min_seq_len,
                max_seq_len=self.config.max_seq_len,
                log_seq_len=self.config.log_seq_len,
                seq_len_per_gp=self.config.seq_len_per_gp,
                min_train_size=self.config.min_train_size,
                max_train_size=self.config.max_train_size,
                replay_small=self.config.replay_small,
                prior_type=self.config.prior_type,
                device=self.config.prior_device,
                n_jobs=1,
            )
        else:
            dataset = LoadPriorDataset(
                data_dir=self.config.prior_dir,
                batch_size=self.config.batch_size,
                ddp_world_size=self.ddp_world_size,
                ddp_rank=self.ddp_rank,
                start_from=self.config.load_prior_start,
                delete_after_load=self.config.delete_after_load,
                device=self.config.prior_device,
            )

        if self.master_process:
            print(dataset)

        self.dataloader = DataLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=0,
            prefetch_factor=None,
            pin_memory=True if self.config.prior_device == "cpu" else False,
            pin_memory_device=self.config.device if self.config.prior_device == "cpu" else "",
        )

    def configure_optimizer(self):
        self.optimizer = optim.AdamW(
            params=self.raw_model.adapter.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = get_scheduler(config=self.config, optimizer=self.optimizer)

    def configure_amp(self):
        self.amp = self.config.amp and "cuda" in self.config.device
        self.scaler = torch.GradScaler("cuda", enabled=self.amp)
        if self.amp:
            if self.master_process:
                print("Automatic Mixed Precision is enabled.")
            self.amp_ctx = torch.autocast(
                device_type="cuda", dtype=torch.float16 if self.config.dtype == "float16" else torch.float32
            )
        else:
            self.amp_ctx = nullcontext()

    def get_latest_checkpoint(self):
        ckpt_dir = self.config.checkpoint_dir
        if not os.path.isdir(ckpt_dir):
            return None

        checkpoints = [f for f in os.listdir(ckpt_dir) if f.startswith("step-") and f.endswith(".ckpt")]
        if not checkpoints:
            return None

        try:
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0]))[-1]
            return os.path.join(ckpt_dir, latest_checkpoint)
        except Exception as e:
            print(f"Error parsing checkpoint filenames: {e}")
            return None

    def load_checkpoint(self):
        checkpoint_path = None
        if hasattr(self.config, "checkpoint_path") and self.config.checkpoint_path:
            checkpoint_path = self.config.checkpoint_path
        elif hasattr(self.config, "checkpoint_dir") and self.config.checkpoint_dir:
            checkpoint_path = self.get_latest_checkpoint()

        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            print("No checkpoint found, starting from scratch.")
            return

        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)

        adapter_state = checkpoint.get("adapter_state_dict")
        if adapter_state is None:
            # backward compatible: full state_dict
            adapter_state = {k.replace("adapter.", ""): v for k, v in checkpoint.get("state_dict", {}).items() if k.startswith("adapter.")}

        if not adapter_state:
            raise ValueError("Checkpoint does not contain adapter weights")

        self.raw_model.adapter.load_state_dict(adapter_state, strict=True)

        if self.config.only_load_model:
            print("Only loading adapter weights")
        else:
            if "optimizer_state" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            if "scheduler_state" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            self.curr_step = int(checkpoint.get("curr_step", 0))
            print(f"Resuming training at step {self.curr_step}")

    def save_checkpoint(self, name: str):
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.config.checkpoint_dir, name)
        checkpoint = {
            "adapter_state_dict": self.raw_model.adapter.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "curr_step": self.curr_step,
        }
        torch.save(checkpoint, checkpoint_path)

    def manage_checkpoint(self):
        ckpt_dir = self.config.checkpoint_dir
        limit = self.config.max_checkpoints

        checkpoints = [f for f in os.listdir(ckpt_dir) if f.startswith("step-") and f.endswith(".ckpt")]
        temp_checkpoints = []
        for ckpt in checkpoints:
            try:
                step = int(ckpt.split("-")[1].split(".")[0])
                if step % self.config.save_perm_every != 0:
                    temp_checkpoints.append((step, ckpt))
            except Exception:
                continue

        temp_checkpoints.sort(key=lambda x: x[0])
        num_to_delete = len(temp_checkpoints) - limit
        if num_to_delete > 0:
            for _, ckpt_name in temp_checkpoints[:num_to_delete]:
                ckpt_path = os.path.join(ckpt_dir, ckpt_name)
                try:
                    os.remove(ckpt_path)
                except Exception as e:
                    print(f"Error removing checkpoint {ckpt_path}: {e}")

    @ddp_cleanup
    def train(self):
        if self.master_process:
            step_progress = tqdm(range(self.curr_step, self.config.max_steps), desc="Step", leave=True)
        else:
            step_progress = range(self.curr_step, self.config.max_steps)

        dataloader = iter(self.dataloader)
        for step in step_progress:
            with Timer() as prior_timer:
                batch = next(dataloader)
            prior_time = prior_timer.elapsed

            with Timer() as train_timer:
                results = self.run_batch(batch)
            train_time = train_timer.elapsed

            torch.cuda.empty_cache()

            self.curr_step = step + 1
            if self.master_process:
                results.update({"prior_time": prior_time, "train_time": train_time})
                step_progress.set_postfix(**{k: round(v, 3) if isinstance(v, float) else v for k, v in results.items()})

                is_temp_save = self.curr_step % self.config.save_temp_every == 0
                is_perm_save = self.curr_step % self.config.save_perm_every == 0
                if is_temp_save or is_perm_save:
                    ckpt_name = f"step-{self.curr_step}.ckpt"
                    self.save_checkpoint(name=ckpt_name)
                    if is_temp_save and not is_perm_save and self.config.max_checkpoints > 0:
                        self.manage_checkpoint()

            if self.wandb_run is not None:
                results["lr"] = self.scheduler.get_last_lr()[0]
                wandb.log(results, step=self.curr_step)

    def validate_micro_batch(self, micro_seq_len, micro_train_size):
        if len(torch.unique(micro_seq_len)) > 1:
            raise ValueError("All datasets in the micro batch must have the same sequence length.")
        if len(torch.unique(micro_train_size)) > 1:
            raise ValueError("All datasets in the micro batch must have the same training size.")
        seq_len = micro_seq_len[0].item()
        train_size = micro_train_size[0].item()
        return seq_len, train_size

    def align_micro_batch(self, micro_X, micro_y, micro_d, seq_len):
        if micro_X.shape[1] > seq_len:
            micro_X = micro_X[:, :seq_len]
        if micro_y.shape[1] > seq_len:
            micro_y = micro_y[:, :seq_len]

        max_features = micro_d.max().item()
        if micro_X.shape[-1] > max_features:
            micro_X = micro_X[..., :max_features]
        return micro_X, micro_y

    def run_micro_batch(self, micro_batch, micro_batch_idx, num_micro_batches):
        micro_X, micro_y, micro_d, micro_seq_len, micro_train_size = micro_batch
        seq_len, train_size = self.validate_micro_batch(micro_seq_len, micro_train_size)
        micro_X, micro_y = self.align_micro_batch(micro_X, micro_y, micro_d, seq_len)

        micro_X = micro_X.to(self.config.device)
        micro_y = micro_y.to(self.config.device)
        micro_d = micro_d.to(self.config.device)

        y_train = micro_y[:, :train_size]
        y_test = micro_y[:, train_size:]

        if self.ddp:
            self.model.require_backward_grad_sync = micro_batch_idx == num_micro_batches - 1

        with self.amp_ctx:
            pred = self.model(micro_X, y_train, micro_d)
            pred = pred.flatten(end_dim=-2)
            true = y_test.long().flatten()
            loss = F.cross_entropy(pred, true)

        scaled_loss = loss / num_micro_batches
        self.scaler.scale(scaled_loss).backward()

        with torch.no_grad():
            micro_results = {"ce": scaled_loss.item()}
            accuracy = (pred.argmax(dim=1) == true).sum() / len(true)
            micro_results["accuracy"] = accuracy.item() / num_micro_batches

        return micro_results

    def run_batch(self, batch):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        batch = [t.to_padded_tensor(padding=0.0) if t.is_nested else t for t in batch]

        num_micro_batches = math.ceil(self.config.batch_size / self.config.micro_batch_size)
        micro_batches = [torch.split(t, self.config.micro_batch_size, dim=0) for t in batch]
        micro_batches = list(zip(*micro_batches))

        results = {"ce": 0.0, "accuracy": 0.0}
        failed_batches = 0

        for idx, micro_batch in enumerate(micro_batches):
            try:
                micro_results = self.run_micro_batch(micro_batch, idx, num_micro_batches)
                for k, v in micro_results.items():
                    results[k] += v
            except torch.cuda.OutOfMemoryError:
                print(
                    f"Warning: OOM error in micro-batch {idx+1}/{num_micro_batches} at step {self.curr_step}. Skipping."
                )
                torch.cuda.empty_cache()
                failed_batches += 1
                continue

        if failed_batches < num_micro_batches:
            if self.config.gradient_clipping > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.raw_model.adapter.parameters(), self.config.gradient_clipping)

            self.scaler.step(self.optimizer)
            self.scaler.update()

        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

        return results


def build_parser_adapter_only():
    parser = build_parser()

    parser.add_argument(
        "--mantis_ckpt",
        type=str,
        default="/data0/fangjuntao2025/tabicl-main/mantis512checkpoints/CaukerImpro-data100k_emb512_100epochs.pt",
        help="Path to Mantis checkpoint (.pt or pretrained dir).",
    )
    parser.add_argument(
        "--tabicl_ckpt",
        type=str,
        default="/data0/fangjuntao2025/tabicl-main/tabICLOrignCheckpoint/tabicl-classifier-v1.1-0506.ckpt",
        help="Path to TabICL checkpoint (.ckpt) that contains icl_predictor weights.",
    )

    parser.add_argument("--mantis_hidden_dim", type=int, default=512, help="Mantis hidden_dim used to instantiate encoder")
    parser.add_argument("--mantis_seq_len", type=int, default=512, help="Mantis seq_len used to instantiate encoder")
    parser.add_argument("--mantis_batch_size", type=int, default=64, help="Batch size used inside Mantis forward")

    parser.add_argument("--adapter_hidden_dim", type=int, default=None, help="Hidden dim of adapter MLP (default=icl_dim)")
    parser.add_argument("--adapter_dropout", type=float, default=0.0, help="Dropout used inside adapter")
    parser.add_argument("--adapter_no_layernorm", action="store_true", help="Disable LayerNorm inside adapter")

    return parser


if __name__ == "__main__":
    print("Starting Mantis+Adapter+ICL adapter-only training...")
    parser = build_parser_adapter_only()
    config = parser.parse_args()

    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    trainer = Trainer(config)
    trainer.train()
