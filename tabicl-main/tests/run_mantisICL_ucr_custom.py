#!/usr/bin/env python3
"""
Instantiate a MantisICL model using:
 - the Mantis pretrained weights from `--mantis-pretrained`
 - the ICL configuration extracted from a TabICL checkpoint `--tabicl-ckpt`

Then evaluate on UCR datasets using the project's `DataReader`.

This script keeps the evaluation simple: it constructs the model, loads pretrained mantis
weights, sets eval mode, and for each UCR dataset performs a single forward pass and
computes accuracy by mapping predicted indices back to original labels via LabelEncoder.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time
import sys

import numpy as np
import torch

from tabicl.model.mantisICL import MantisICL
from tabicl.prior.data_reader import DataReader
from sklearn.preprocessing import LabelEncoder


def load_icl_from_checkpoint(mantis: MantisICL, ckpt_path: Path):
    """Try to extract ICL predictor params from a TabICL checkpoint state_dict and load
    them into mantis. Returns (missing_keys, loaded_keys).
    The function attempts several common prefix variants used in checkpoints.
    """
    import torch

    ck = torch.load(str(ckpt_path), map_location='cpu')
    sd = ck.get('state_dict', ck if isinstance(ck, dict) else {})
    if not isinstance(sd, dict):
        raise RuntimeError("Checkpoint state_dict not found or invalid")

    target_state = mantis.icl_predictor.state_dict()
    mapped = {}

    # for quick lookup
    sd_keys = set(sd.keys())

    for key in target_state.keys():
        # candidate patterns to try
        candidates = [
            f"model_.icl_predictor.{key}",
            f"model.icl_predictor.{key}",
            f"icl_predictor.{key}",
            f"model_.icl.{key}",
            f"model.icl.{key}",
            f"icl.{key}",
            key,
        ]
        found = False
        for c in candidates:
            if c in sd:
                mapped[key] = sd[c]
                found = True
                break

        if not found:
            # fallback: find any sd key that endswith the target key
            for sdk in sd_keys:
                if sdk.endswith('.' + key) or sdk.endswith('/' + key):
                    mapped[key] = sd[sdk]
                    found = True
                    break

    # load into module (non-strict to allow partial matches)
    try:
        missing = [k for k in target_state.keys() if k not in mapped]
        if mapped:
            mantis.icl_predictor.load_state_dict(mapped, strict=True)
        return missing, list(mapped.keys())
    except Exception as e:
        raise RuntimeError(f"Failed to load icl predictor weights: {e}")


def load_ckpt_config(ckpt_path: Path):
    import torch

    ck = torch.load(str(ckpt_path), map_location='cpu')
    if not isinstance(ck, dict) or 'config' not in ck:
        raise RuntimeError(f"Invalid checkpoint or missing 'config' in {ckpt_path}")
    return ck['config']


def evaluate_mantis(mantis: MantisICL, reader: DataReader, max_datasets: int | None = None, outdir: Path | None = None):
    results = []
    ds_list = reader.dataset_list_ucr
    if max_datasets is not None:
        ds_list = ds_list[:max_datasets]

    mantis.eval()
    device = next(mantis.parameters()).device

    for name in ds_list:
        try:
            X_train, y_train = reader.read_dataset(name, which_set='train')
            X_test, y_test = reader.read_dataset(name, which_set='test')

            # 确保数据形状为 (n_samples, seq_len)
            if len(X_train.shape) == 3:
                X_train = X_train.squeeze(1)
            if len(X_test.shape) == 3:
                X_test = X_test.squeeze(1)

            # 转换为 Tensor
            X_train = torch.from_numpy(X_train).float().to(device)
            y_train = torch.from_numpy(y_train).long().to(device)
            X_test = torch.from_numpy(X_test).float().to(device)
            y_test = torch.from_numpy(y_test).long().to(device)
            X = torch.cat([X_train.unsqueeze(0), X_test.unsqueeze(0)], dim=1)
            train_size = y_train.shape[0]

            # 使用模型的 forward 方法进行推理
            with torch.no_grad():
                # 修改后
                logits = mantis(
                    X,
                    y_train.unsqueeze(0),  # y_train: (B=1, train_size)
                    embed_with_test=False,
                    return_logits=True,
                )
            # 提取测试集的预测结果
            preds = logits.argmax(dim=-1).squeeze(0)  # (test_size,)
            acc = (preds == y_test).float().mean().item()
            results.append((name, acc))
            logging.info(f"{name}: acc={acc:.4f}")
            # results.append((name, acc))
            # # if X_train.ndim == 3:
            #     X_train = X_train.squeeze(1)
            # if X_test.ndim == 3:
            #     X_test = X_test.squeeze(1)

            # # label encoding
            # le = LabelEncoder()
            # y_train_enc = le.fit_transform(y_train)

            # # tensors
            # Xt = torch.from_numpy(X_train).float().to(device)
            # Xs = torch.from_numpy(X_test).float().to(device)
            # yt = torch.from_numpy(y_train_enc).long().to(device)

            # # construct concatenated input: (B=1, T=train+test, H)
            # X_cat = torch.cat([Xt.unsqueeze(0), Xs.unsqueeze(0)], dim=1)

            # with torch.no_grad():
            #     out = mantis(X_cat, yt.unsqueeze(0), embed_with_test=False, return_logits=True)

            # # out expected shape: (B, test_size, n_classes) or (B, T, C)
            # out_np = out.detach().cpu().numpy()
            # B, S, C = out_np.shape
            # train_size = y_train_enc.shape[0]
            # # if S equals test_size, good; if S == T, take slice
            # if S == Xs.shape[0]:
            #     logits_test = out_np[0]
            # elif S == (Xt.shape[0] + Xs.shape[0]):
            #     logits_test = out_np[0, train_size:, :]
            # else:
            #     # best-effort: if S > train_size, assume last (S-train_size) are test
            #     if S > train_size:
            #         logits_test = out_np[0, train_size:, :]
            #     else:
            #         raise RuntimeError(f"Unexpected logits shape {out_np.shape} for dataset {name}")

            # # 检查模型对 test 部分输出的类别数
            # num_output_classes = int(logits_test.shape[-1])
            # if num_output_classes > 10:
            #     logging.warning(f"{name}: model outputs {num_output_classes} classes (>10)")
            #     # 同时在控制台打印一个数值，便于快速检查
            #     print(f"{name}: model outputs {num_output_classes} classes")

            # preds_idx = np.argmax(logits_test, axis=-1)
            # preds = le.inverse_transform(preds_idx)

            # acc = float(np.mean(preds == y_test))
            # logging.info(f"{name}: acc={acc:.4f}")
            # results.append((name, acc))

        except Exception as e:
            logging.exception(f"Failed dataset {name}: {e}")

    avg = None
    if results:
        avg = sum(a for _, a in results) / len(results)

    # persist results and summary
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        with open(outdir / 'mantis_custom_ucr_results.txt', 'w') as f:
            for n, a in results:
                f.write(f"{n}\t{a:.6f}\n")
        if results:
            with open(outdir / 'mantis_custom_ucr_summary.txt', 'w') as f:
                f.write(f"datasets\t{len(results)}\naverage_acc\t{avg:.6f}\n")

    # log and print average accuracy
    if avg is not None:
        logging.info(f"Evaluated {len(results)} datasets. Average accuracy: {avg:.6f}")
        print(f"Evaluated {len(results)} datasets. Average accuracy: {avg:.6f}")

    return results, avg


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--tabicl-ckpt', type=str, default='/data0/fangjuntao2025/tabicl-main/tabICLOrignCheckpoint/tabicl-classifier-v1.1-0506.ckpt')
    p.add_argument('--mantis-pretrained', type=str, default='/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/Models/Mantis/Mantis_cheickpoint')
    p.add_argument('--data-path', type=str, default='/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/data/')
    p.add_argument('--outdir', type=str, default='evaluation_results')
    p.add_argument('--max-datasets', type=int, default=128)
    p.add_argument('--icl-dim', type=int, default=None, help='Optional override for icl_dim (defaults in MantisICL)')
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format='[%(levelname)s] %(message)s')

    ck_cfg = load_ckpt_config(Path(args.tabicl_ckpt))
    # extract icl-related args
    icl_kwargs = {}
    for k in ['max_classes', 'icl_num_blocks', 'icl_nhead', 'ff_factor', 'dropout', 'activation', 'norm_first']:
        if k in ck_cfg:
            icl_kwargs[k] = ck_cfg[k]

    # optionally set icl_dim from CLI
    if args.icl_dim is not None:
        icl_kwargs['icl_dim'] = args.icl_dim

    logging.info('Constructing MantisICL with icl kwargs: %s', icl_kwargs)

    # device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mantis = MantisICL(**icl_kwargs)
    # load mantis pretrained (replace internal mantis model)
    try:
        mantis.mantis_model = mantis.mantis_model.from_pretrained(args.mantis_pretrained)
        logging.info('Loaded mantis pretrained from %s', args.mantis_pretrained)
    except Exception as e:
        logging.exception('Failed to load mantis pretrained: %s', e)

    # attempt to load ICL predictor params from tabicl checkpoint
    try:
        missing, loaded = load_icl_from_checkpoint(mantis, Path(args.tabicl_ckpt))
        logging.info('Loaded %d icl predictor keys from checkpoint, %d missing', len(loaded), len(missing))
        if len(loaded) < 1:
            logging.info('No matching icl predictor keys found in checkpoint state_dict')
    except Exception as e:
        logging.exception('Failed to map/load icl predictor from checkpoint: %s', e)

    mantis.to(device)

    reader = DataReader(data_path=args.data_path)

    outdir = Path(args.outdir)
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_sub = outdir / f'mantis_custom_ucr_{ts}'

    results, avg = evaluate_mantis(mantis, reader, max_datasets=args.max_datasets, outdir=out_sub)
    if avg is not None:
        logging.info('Final average accuracy across evaluated datasets: %.6f', avg)
        print('Final average accuracy across evaluated datasets: %.6f' % avg)


if __name__ == '__main__':
    main()
