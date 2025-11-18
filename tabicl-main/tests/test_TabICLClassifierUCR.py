import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tabicl.sklearn.classifier import TabICLClassifier

import pandas as pd
import os
import multiprocessing as mp
from tabicl.prior.data_reader import DataReader

MANTIS_CHECKPOINT ="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/Models/Mantis/checkpoint/CaukerMixed-data100k_200_2e-3_100epochs.pt"
TABICL_CHECKPOINT = "/data0/fangjuntao2025/tabicl-main/tabICLOrignCheckpoint/tabicl-classifier-v1.1-0506.ckpt"
DEFAULT_UEA_PATH = "/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UEAData/"
DEFAULT_UCR_PATH = "/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UCRdata/"
BATCHSIZE =   32
MAXWORKERS = 3
# Module-level worker helpers for multiprocessing (must be picklable / top-level)
WORKER_READER = None
WORKER_CLF = None
WORKER_USE_MANTIS = False
USE_PARALLEL_EVAL = os.environ.get("TABICL_USE_MULTIGPU", "1") == "1"
# USE_PARALLEL_EVAL = 0

SKIP_UEA_DATASETS = {
    "AtrialFibrillation",
    "BasicMotions",
    "Cricket",
    "ArticularyWordRecognition",
    "ERing",
    "CharacterTrajectories",
    "EigenWorms",
    "Epilepsy",
    "EthanolConcentration",
    "FingerMovements",
    "HandMovementDirection",
    "Handwriting",
    "Heartbeat",
    "JapaneseVowels",
    "LSST",
    "Libras",
    "NATOPS",
    "MotorImagery",
    "PenDigits",
    "RacketSports",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "StandWalkJump",
    "UWaveGestureLibrary",
    "PhonemeSpectra",
}

def _prepare_feature_array(array: np.ndarray) -> np.ndarray:
    """Ensure arrays fed into TabICLClassifier are 2D."""
    if array.ndim <= 2:
        return array
    # common case: (N, 1, L) -> squeeze the singleton channel axis
    if array.ndim == 3 and array.shape[1] == 1:
        return array.squeeze(1)
    # otherwise flatten all trailing axes into feature dimension
    return array.reshape(array.shape[0], -1)

def _worker_init(uea_path, ucr_path, use_mantis, mantis_batch_size, tabicl_ckpt, mantis_ckpt, transform_ts_size):
    """Initializer run once per worker process to create heavy objects and bind a GPU."""
    global WORKER_READER, WORKER_CLF, WORKER_USE_MANTIS

    # assign a GPU to this worker based on its rank and CUDA_VISIBLE_DEVICES
    try:
        gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
        # multiprocessing sets _identity to a 1-based worker id
        identity = mp.current_process()._identity
        if identity:
            rank = identity[0] - 1  # 0-based
            gpu_idx = int(gpu_ids[rank % len(gpu_ids)])
            if torch.cuda.is_available():
                torch.cuda.set_device(gpu_idx)
    except Exception:
        # if anything goes wrong, just fall back to default device
        pass

    WORKER_READER = DataReader(UEA_data_path=uea_path, UCR_data_path=ucr_path, transform_ts_size=transform_ts_size)
    WORKER_USE_MANTIS = use_mantis
    clf_kwargs = dict(
        verbose=False,
        n_estimators=32,
        checkpoint_version="tabicl-classifier-v1.1-0506.ckpt",
        model_path=tabicl_ckpt,
    )
    if use_mantis:
        clf_kwargs.update(mantis_checkpoint=mantis_ckpt, mantis_batch_size=mantis_batch_size)
    WORKER_CLF = TabICLClassifier(**clf_kwargs)


def _worker_eval(dataset_name):
    """Evaluate a single dataset inside a worker process; returns (name, acc, err)."""
    global WORKER_READER, WORKER_CLF, WORKER_USE_MANTIS
    try:
        X_train, y_train = WORKER_READER.read_dataset(dataset_name, which_set='train')
        X_test, y_test = WORKER_READER.read_dataset(dataset_name, which_set='test')
        # 当 worker 未使用 mantis 时，将 3D [N, C, L] 展平为 2D [N, C*L]
        if not WORKER_USE_MANTIS:
            X_train = _prepare_feature_array(X_train)
            X_test = _prepare_feature_array(X_test)
        WORKER_CLF.fit(X_train, y_train)
        y_pred = WORKER_CLF.predict(X_test)
        acc = float(np.mean(y_pred == y_test))
        print(f"{dataset_name} accuracy: {acc:.4f}")
        return (dataset_name, acc, None)
    except torch.cuda.OutOfMemoryError as oom:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        msg = f"CUDA OOM for {dataset_name}: {oom}"
        print(msg)
        return (dataset_name, 0.0, msg)
    except Exception as e:
        return (dataset_name, 0.0, str(e))


def evaluate_datasets_parallel(dataset_names, worker_config, max_workers=MAXWORKERS):
    """Evaluate datasets across multiple GPUs/processes using the shared worker helpers."""
    if not dataset_names:
        return []

    requested = min(max_workers, len(dataset_names), mp.cpu_count() or 1)
    requested = max(1, requested)
    ctx = mp.get_context('spawn')
    results = []

    initargs = (
        worker_config['uea_path'],
        worker_config['ucr_path'],
        worker_config['use_mantis'],
        worker_config['mantis_batch_size'],
        worker_config['tabicl_ckpt'],
        worker_config['mantis_ckpt'],
        worker_config['transform_ts_size'],
    )

    with ctx.Pool(processes=requested, initializer=_worker_init, initargs=initargs) as pool:
        for name, acc, err in pool.imap_unordered(_worker_eval, dataset_names):
            if err:
                print(f"Dataset {name} failed in worker: {err}")
                continue
            results.append((name, acc))

    return sorted(results)



class UCR_UEAEvaluator:
    def __init__(self,
                 UEA_data_path: str = "/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UEAData/",
                 UCR_data_path: str = "/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UCRdata/",
                 use_mantis: bool = True,
                 mantis_batch_size: int = 64,
                 log_processing: bool = False):
        """
        初始化评估器。

        Parameters
        ----------
        UEA_data_path : str
            UEA 数据集的路径。
        UCR_data_path : str
            UCR 数据集的路径。
        use_mantis : bool, default=True
            是否在 TabICLClassifier 中启用 Mantis 预编码。
        mantis_batch_size : int, default=64
            Mantis 编码的微批量大小。
        log_processing : bool, default=False
            是否在 DataReader 读取数据时打印数据集名称、NaN 处理方式和插值长度。
        """
        # use the provided dataset paths when constructing the DataReader
        self.reader = DataReader(
            UEA_data_path=UEA_data_path,
            UCR_data_path=UCR_data_path,
            transform_ts_size=512,
            log_processing=log_processing,
        )
        self.results = []
        self.stats = []
        self.use_mantis = use_mantis
        self.mantis_batch_size = mantis_batch_size
        # Instantiate classifier once to avoid re-loading the heavy checkpoint repeatedly.
        classifier_kwargs = dict(
            verbose=False,
            n_estimators=32,
            checkpoint_version="tabicl-classifier-v1.1-0506.ckpt",
            model_path=TABICL_CHECKPOINT,
        )
        if self.use_mantis:
            classifier_kwargs.update(
                mantis_checkpoint=MANTIS_CHECKPOINT,
                mantis_batch_size=self.mantis_batch_size,
            )
        # keep a single classifier instance; fit() will overwrite dataset-specific
        # encoders/ensemble but the underlying model weights will stay loaded.
        self.clf = TabICLClassifier(**classifier_kwargs)
        # store config for possible worker reuse / parallel init
        # store config for possible worker reuse / parallel init
        self._worker_config = dict(
            uea_path=UEA_data_path,
            ucr_path=UCR_data_path,
            use_mantis=self.use_mantis,
            mantis_batch_size=self.mantis_batch_size,
            tabicl_ckpt=TABICL_CHECKPOINT,
            mantis_ckpt=MANTIS_CHECKPOINT,
            transform_ts_size=512,
        )

    def evaluate_dataset(self, dataset_name: str) -> float:
        """
        评估单个数据集。

        Parameters
        ----------
        dataset_name : str
            数据集名称。

        Returns
        -------
        float
            该数据集上的准确率。
        """
        try:
            # 加载数据
            X_train, y_train = self.reader.read_dataset(dataset_name, which_set='train')   
            X_test, y_test = self.reader.read_dataset(dataset_name, which_set='test')

            # 如果未使用 mantis，则将 3D [N, C, L] 展平为 2D [N, C*L]
            if not self.use_mantis:
                X_train = _prepare_feature_array(X_train)
                X_test = _prepare_feature_array(X_test)

            dataset_stats = self._collect_stats(dataset_name, X_train, X_test)
            self.stats.append(dataset_stats)

            # 初始化并训练分类器
            # reuse pre-instantiated classifier to avoid heavyweight reloads
            clf = self.clf
            clf.fit(X_train, y_train)

            # 预测并计算准确率
            y_pred = clf.predict(X_test)
            accuracy = np.mean(y_pred == y_test)

            return accuracy

        except torch.cuda.OutOfMemoryError as oom:
            print(f"[SKIP] {dataset_name} -> CUDA OOM: {oom}")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return 0.0
        except Exception as e:
            print(f"Failed to evaluate dataset {dataset_name}: {e}")
            return 0.0

    def _collect_stats(self, dataset_name: str, X_train: np.ndarray, X_test: np.ndarray) -> dict:
        """统计训练集与测试集的全局数值特征。"""

        def summarize(array: np.ndarray) -> dict:
            return {
                "min": float(np.nanmin(array)),
                "max": float(np.nanmax(array)),
                "mean": float(np.nanmean(array)),
                "std": float(np.nanstd(array)),
            }

        return {
            "dataset": dataset_name,
            "train_shape": tuple(map(int, X_train.shape)),
            "test_shape": tuple(map(int, X_test.shape)),
            "train": summarize(X_train),
            "test": summarize(X_test),
        }

    def evaluate_all(self) -> None:
        """
        评估所有 UCR 和 UEA 数据集，分别计算在两类集合上的平均准确率。
        """
        print("Starting evaluation on UCR and UEA datasets...")

        ucr_results = []
        uea_results = []

        # Parallel evaluation using process pool: each worker will initialize its own
        # DataReader and TabICLClassifier once (see module-level _worker_init/_worker_eval).
        # To reduce GPU OOM risk, cap the number of workers.
        max_workers = MAXWORKERS
        n_jobs = min(max_workers, mp.cpu_count() or 1, max(1, (len(self.reader.dataset_list_ucr) + len(self.reader.dataset_list_uea)) // 2))
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=n_jobs, initializer=_worker_init, initargs=(
            self._worker_config['uea_path'],
            self._worker_config['ucr_path'],
            self._worker_config['use_mantis'],
            self._worker_config['mantis_batch_size'],
            self._worker_config['tabicl_ckpt'],
            self._worker_config['mantis_ckpt'],
            self._worker_config['transform_ts_size'],
        )) as pool:
            # Evaluate UCR
            # ucr_list = sorted(self.reader.dataset_list_ucr)
            # for name, acc, err in pool.imap_unordered(_worker_eval, ucr_list):
            #     if err:
            #         print(f"UCR {name} failed: {err}")
            #     elif acc > 0:
            #         ucr_results.append((name, acc))
            #         self.results.append((name, acc))
            #         print(f"UCR {name}: {acc:.4f}")

            # Evaluate UEA
            uea_list = sorted(self.reader.dataset_list_uea)
            for name, acc, err in pool.imap_unordered(_worker_eval, uea_list):
                if err:
                    print(f"UEA {name} failed: {err}")
                elif acc > 0:
                    uea_results.append((name, acc))
                    self.results.append((name, acc))
                    print(f"UEA {name}: {acc:.4f}")

        # Summaries
        if ucr_results:
            ucr_avg = sum(v for _, v in ucr_results) / len(ucr_results)
            print(f"\nUCR: evaluated {len(ucr_results)} datasets, average accuracy: {ucr_avg:.4f}")
            results_dir = Path("evaluation_results")
            results_dir.mkdir(exist_ok=True)
            with open(results_dir / "ucr_detailed.txt", "w") as f:
                for name, val in ucr_results:
                    f.write(f"{name}: {val:.6f}\n")
            with open(results_dir / "ucr_summary.txt", "w") as f:
                f.write(f"Total datasets: {len(ucr_results)}\n")
                f.write(f"Average accuracy: {ucr_avg:.6f}\n")
        else:
            print("No UCR datasets were successfully evaluated.")

        if uea_results:
            uea_avg = sum(v for _, v in uea_results) / len(uea_results)
            print(f"\nUEA: evaluated {len(uea_results)} datasets, average accuracy: {uea_avg:.4f}")
            results_dir = Path("evaluation_results")
            results_dir.mkdir(exist_ok=True)
            with open(results_dir / "uea_detailed.txt", "w") as f:
                for name, val in uea_results:
                    f.write(f"{name}: {val:.6f}\n")
            with open(results_dir / "uea_summary.txt", "w") as f:
                f.write(f"Total datasets: {len(uea_results)}\n")
                f.write(f"Average accuracy: {uea_avg:.6f}\n")
        else:
            print("No UEA datasets were successfully evaluated.")

        # also save overall results with existing save_results for backward compat
        self.save_results()

    def save_results(self) -> None:
        """
        保存评估结果到文件。
        """
        results_dir = Path("evaluation_results")
        results_dir.mkdir(exist_ok=True)
        
        # 保存详细结果
        with open(results_dir / "detailed_results.txt", "w") as f:
            for dataset_name, accuracy in self.results:
                f.write(f"{dataset_name}: {accuracy:.4f}\n")
        
        # 保存汇总结果（防止空结果时除零）
        with open(results_dir / "summary.txt", "w") as f:
            total = len(self.results)
            f.write(f"Total datasets: {total}\n")
            if total > 0:
                avg_accuracy = sum(acc for _, acc in self.results) / total
                f.write(f"Average accuracy: {avg_accuracy:.4f}\n")
            else:
                f.write("Average accuracy: N/A (no successful evaluations)\n")

        with open(results_dir / "ucr_stats.jsonl", "w") as f:
            for stat in self.stats:
                record = {
                    "dataset": stat["dataset"],
                    "train_shape": list(stat["train_shape"]),
                    "test_shape": list(stat["test_shape"]),
                    "train_min": stat["train"]["min"],
                    "train_max": stat["train"]["max"],
                    "train_mean": stat["train"]["mean"],
                    "train_std": stat["train"]["std"],
                    "test_min": stat["test"]["min"],
                    "test_max": stat["test"]["max"],
                    "test_mean": stat["test"]["mean"],
                    "test_std": stat["test"]["std"],
                }
                f.write(json.dumps(record) + "\n")


def read_ucr_tsv(train_path, test_path):  # 不改变时序数据的长度
    """
    直接读取 UCR 数据集的 TSV 文件。
    train_path: 训练集文件路径
    test_path: 测试集文件路径
    返回: X_train, y_train, X_test, y_test
    """
    train_df = pd.read_csv(train_path, sep='\t', header=None)
    y_train = train_df.iloc[:, 0].values
    X_train = train_df.iloc[:, 1:].values
    test_df = pd.read_csv(test_path, sep='\t', header=None)
    y_test = test_df.iloc[:, 0].values
    X_test = test_df.iloc[:, 1:].values
    return X_train, y_train, X_test, y_test
    
def batch_test_ucr(ucr_root, use_mantis: bool = True):
    """
    批量读取 UCR 路径下所有数据集并评测。
    ucr_root: UCRArchive_2018 路径（如 /data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/data/UCRArchive_2018）
    """
    results = []
    # 只初始化一次模型
    clf_kwargs = dict(
        verbose=False,
        checkpoint_version="tabicl-classifier-v1.1-0506.ckpt",
        model_path=TABICL_CHECKPOINT,
    )
    if use_mantis:
        clf_kwargs.update(
            mantis_checkpoint=MANTIS_CHECKPOINT,
            mantis_batch_size=BATCHSIZE,
        )

    clf = TabICLClassifier(**clf_kwargs)
    for dataset_name in sorted(os.listdir(ucr_root)):
        dataset_dir = os.path.join(ucr_root, dataset_name)
        train_path = os.path.join(dataset_dir, f"{dataset_name}_TRAIN.tsv")
        test_path = os.path.join(dataset_dir, f"{dataset_name}_TEST.tsv")
        if not (os.path.isfile(train_path) and os.path.isfile(test_path)):
            print(f"跳过无效数据集: {dataset_name}")
            continue
        try:
            X_train, y_train, X_test, y_test = read_ucr_tsv(train_path, test_path)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = np.mean(y_pred == y_test)
            print(f"{dataset_name}: {accuracy:.4f}")
            results.append((dataset_name, accuracy))
        except Exception as e:
            print(f"{dataset_name} 测试失败: {e}")
    if results:
        avg_acc = sum(acc for _, acc in results) / len(results)
        print(f"\n共评测 {len(results)} 个数据集，平均准确率: {avg_acc:.4f}")
        results_dir = Path("evaluation_results")
        results_dir.mkdir(exist_ok=True)
        with open(results_dir / "ucr_batch_detailed.txt", "w") as f:
            for dataset_name, accuracy in results:
                f.write(f"{dataset_name}: {accuracy:.4f}\n")
        with open(results_dir / "ucr_batch_summary.txt", "w") as f:
            f.write(f"Total datasets: {len(results)}\n")
            f.write(f"Average accuracy: {avg_acc:.4f}\n")
    else:
        print("无有效数据集被评测！")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate TabICL on UEA/UCR datasets with optional CLI overrides."
    )
    parser.add_argument(
        "--uea-path",
        default=DEFAULT_UEA_PATH,
        help=f"UEA dataset root directory (default: {DEFAULT_UEA_PATH})",
    )
    parser.add_argument(
        "--ucr-path",
        default=DEFAULT_UCR_PATH,
        help=f"UCR dataset root directory (default: {DEFAULT_UCR_PATH})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluator = UCR_UEAEvaluator(
        UEA_data_path=args.uea_path,
        UCR_data_path=args.ucr_path,
        mantis_batch_size=BATCHSIZE,
        use_mantis=WORKER_USE_MANTIS,
        log_processing=True,
    )
    # target_datasets = [
    #     "BasicMotions",
    #     "DuckDuckGeese",
    #     "InsectWingbeat",
    #     "FaceDetection",
    #     "SpokenArabicDigits",
    #     "PEMS-SF",
    # ]
    # 1) 遍历所有 UEA 数据集，逐个评测并记录结果
    dataset_names = sorted(evaluator.reader.dataset_list_uea)
    # dataset_names = [
    #     name for name in sorted(evaluator.reader.dataset_list_uea)
    #     if name not in SKIP_UEA_DATASETS
    # ]
    all_uea_results = []

    if USE_PARALLEL_EVAL:
        print("\n===== Evaluating ALL UEA datasets with multiprocessing / multi-GPU =====")
        parallel_results = evaluate_datasets_parallel(dataset_names, evaluator._worker_config, max_workers=MAXWORKERS)
        for name, acc in parallel_results:
            print(f"{name} accuracy: {acc:.4f}")
            all_uea_results.append((name, float(acc)))
            evaluator.results.append((name, float(acc)))
    else:
        print("\n===== Evaluating ALL UEA datasets (single-process) =====")
        for name in dataset_names:
            print(f"\n----- Evaluating {name} -----")
            acc = evaluator.evaluate_dataset(name)
            print(f"{name} accuracy: {acc:.4f}")
            try:
                acc_val = float(acc)
            except Exception:
                acc_val = 0.0
            if acc_val > 0:
                all_uea_results.append((name, acc_val))
                evaluator.results.append((name, acc_val))

    # 2) 计算所有 UEA 数据集的平均准确率并保存
    results_dir = Path("evaluation_results")
    results_dir.mkdir(exist_ok=True)
    if all_uea_results:
        avg_acc = sum(v for _, v in all_uea_results) / len(all_uea_results)
        print(f"\nAll UEA datasets evaluated: {len(all_uea_results)}, average accuracy: {avg_acc:.4f}")

        with open(results_dir / "TabICL_uea_all_detailed.txt", "w") as f:
            for name, val in all_uea_results:
                f.write(f"{name}: {val:.6f}\n")

        with open(results_dir / "TabICL_uea_all_summary.txt", "w") as f:
            f.write(f"Total datasets: {len(all_uea_results)}\n")
            f.write(f"Average accuracy: {avg_acc:.6f}\n")
    else:
        print("No UEA datasets were successfully evaluated.")

    # # collect results for these selected datasets and compute average accuracy
    # selected_results = []
    # for name in target_datasets:
    #     print(f"\n===== Evaluating {name} =====")
    #     acc = evaluator.evaluate_dataset(name)
    #     print(f"{name} accuracy: {acc:.4f}")
    #     # record successful results (acc > 0)
    #     try:
    #         acc_val = float(acc)
    #     except Exception:
    #         acc_val = 0.0
    #     if acc_val > 0:
    #         selected_results.append((name, acc_val))
    #         # also append to evaluator.results for compatibility with save_results
    #         evaluator.results.append((name, acc_val))

    # # compute and persist average accuracy over the selected datasets
    # results_dir = Path("evaluation_results")
    # results_dir.mkdir(exist_ok=True)
    # if selected_results:
    #     avg_acc = sum(v for _, v in selected_results) / len(selected_results)
    #     print(f"\nSelected UEA datasets evaluated: {len(selected_results)}, average accuracy: {avg_acc:.4f}")
    #     with open(results_dir / "selected_uea_detailed.txt", "w") as f:
    #         for name, val in selected_results:
    #             f.write(f"{name}: {val:.6f}\n")
    #     with open(results_dir / "selected_uea_summary.txt", "w") as f:
    #         f.write(f"Total datasets: {len(selected_results)}\n")
    #         f.write(f"Average accuracy: {avg_acc:.6f}\n")
    # else:
    #     print("No selected datasets were successfully evaluated.")

    # also save aggregated results/stats via existing helper
    #evaluator.save_results()
    
    
    
    
    
    # 2. 新增：批量读取 UCRArchive_2018 下所有数据集并评测
    # ucr_root = "/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/data/UCRArchive_2018"
    # print("\n批量评测 UCRArchive_2018 下所有数据集：")
    # batch_test_ucr(ucr_root)
