import torch
from typing import List, Tuple
from tabicl import TabICL  # 假设模型类为 TabICL
from torch import Tensor
from tabicl.prior.data_reader import DataReader
from tabicl.train.train_config import build_parser
from tabicl.model.mantisICL import MantisICL
from pathlib import Path
import logging

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

class UCREvaluator:
    def __init__(self, model: MantisICL, reader: DataReader, device: str = "cuda"):
        """
        初始化评估器。

        Parameters
        ----------
        model : TabICL
            训练好的 tabICL 模型。
        reader : UCRDatasetReader
            UCR 数据集读取器。
        device : str, optional
            设备类型，默认为 "cuda"。
        """
        self.model = model
        self.reader = reader
        self.device = device

    def _load_datasets(self) -> List[Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]]:
        """
        加载所有 UCR 数据集的训练集和测试集。

        Returns
        -------
        List[Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]]
            每个数据集的 (训练集, 测试集) 元组列表。
        """
        datasets = []
        for dataset_name in self.reader.dataset_list_ucr:
            try:
                # 加载训练集
                X_train, y_train = self.reader.read_dataset(dataset_name, which_set='train')
                # 加载测试集
                X_test, y_test = self.reader.read_dataset(dataset_name, which_set='test')

                # 确保数据形状为 (n_samples, seq_len)
                if len(X_train.shape) == 3:
                    X_train = X_train.squeeze(1)
                if len(X_test.shape) == 3:
                    X_test = X_test.squeeze(1)

                # 转换为 Tensor
                X_train = torch.from_numpy(X_train).float().to(self.device)
                y_train = torch.from_numpy(y_train).long().to(self.device)
                X_test = torch.from_numpy(X_test).float().to(self.device)
                y_test = torch.from_numpy(y_test).long().to(self.device)

                datasets.append((dataset_name, (X_train, y_train), (X_test, y_test)))
            except Exception as e:
                print(f"Failed to load dataset {dataset_name}: {e}")
                continue
        return datasets

    def evaluate(self) -> List[Tuple[str, float]]:
        """
        评估模型在所有 UCR 数据集上的表现。

        Returns
        -------
        List[Tuple[str, float]]
            每个数据集的名称和准确率。
        """
        results = []
        datasets = self._load_datasets()
        self.model.eval()

        for dataset_name, (X_train, y_train), (X_test, y_test) in datasets:
            # 将训练集和测试集合并为一个输入张量
            # X: (B=1, T=train_size + test_size, H=seq_len)
            X = torch.cat([X_train.unsqueeze(0), X_test.unsqueeze(0)], dim=1)
            train_size = y_train.shape[0]

            # 使用模型的 forward 方法进行推理
            with torch.no_grad():
                # 修改后
                logits = self.model(
                    X,
                    y_train.unsqueeze(0),  # y_train: (B=1, train_size)
                    embed_with_test=False,
                    return_logits=True,
                )
            # 提取测试集的预测结果
            preds = logits.argmax(dim=-1).squeeze(0)  # (test_size,)
            accuracy = (preds == y_test).float().mean().item()
            results.append((dataset_name, accuracy))

        return results

    def print_results(self, results: List[Tuple[str, float]]):
        """
        打印评估结果。

        Parameters
        ----------
        results : List[Tuple[str, float]]
            每个数据集的名称和准确率。
        """
        print("UCR Dataset Evaluation Results:")
        print("-----------------------------")
        total_accuracy = 0.0
        for dataset_name, accuracy in results:
            print(f"{dataset_name}: Accuracy = {accuracy:.4f}")
            total_accuracy += accuracy
        avg_accuracy = total_accuracy / len(results) if results else 0.0
        print("-----------------------------")
        print(f"Average Accuracy = {avg_accuracy:.4f}")
        print(f"数据集数量{len(results)}")
        print("-----------------------------")

# 示例用法
if __name__ == "__main__":
    parser = build_parser()
    config = parser.parse_args()
    model_config = {
            "max_classes":  config.max_classes,
            "icl_num_blocks":  config.icl_num_blocks,
            "icl_nhead":  config.icl_nhead,
            "ff_factor":  config.ff_factor,
            "dropout":  config.dropout,
            "activation":  config.activation,
            "norm_first":  config.norm_first,
            "train_mantis":config.train_mantis
    }

    model = MantisICL(**model_config)  # 使用 ** 解包字典参数
    
    # 初始化模型和数据读取器

    reader = DataReader(data_path="/home/hzf00006536/fjt/CauKer/data/")  # 替换为实际数据读取器初始化代码
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 添加多卡支持
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    ###
    # model = model.to(device)  # 将模型移动到指定设备
    #
    # checkpoint = torch.load("/data0/fangjuntao2025/tabicl-main/src/tabicl/checkpointsMantisICL12blocks/step-42300.ckpt", map_location=device, weights_only=True)
    #
    # # Load model state
    # if "state_dict" not in checkpoint:
    #     raise ValueError("Checkpoint does not contain model state")
    # # 修改后
    # state_dict = checkpoint["state_dict"]
    # # 如果模型使用了DataParallel，需要添加module.前缀
    # if isinstance(model, torch.nn.DataParallel):
    #     state_dict = {"module." + k: v for k, v in state_dict.items()}
    # model.load_state_dict(state_dict)
    #####
    model_params = torch.load('/home/hzf00006536/fjt/CauKer/Models/Mantis/checkpoint/CaukerImpro-data100k_emb256_100epochs.pt', weights_only=True)
    model.mantis_model.load_state_dict(model_params)
    #model.mantis_model  =  model.mantis_model.from_pretrained("/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/Models/Mantis/Mantis_cheickpoint")  # 加载预训练模型权重
    try:
        missing, loaded = load_icl_from_checkpoint(model, Path('/home/hzf00006536/fjt/tabicl-main/tabicl-main/src/tabicl/checkpointsTabICL/tabicl-classifier-v1.1-0506.ckpt'))
        logging.info('Loaded %d icl predictor keys from checkpoint, %d missing', len(loaded), len(missing))
        if len(loaded) < 1:
            logging.info('No matching icl predictor keys found in checkpoint state_dict')
    except Exception as e:
        logging.exception('Failed to map/load icl predictor from checkpoint: %s', e)
    model = model.to(device)
    #####
    print(f"Model device: {next(model.parameters()).device}")
    # 初始化评估器
    evaluator = UCREvaluator(model, reader, device)

    # 评估并打印结果
    results = evaluator.evaluate()
    evaluator.print_results(results)
