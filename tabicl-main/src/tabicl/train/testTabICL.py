import torch
from typing import List, Tuple
from tabicl import TabICL  # 假设模型类为 TabICL
from torch import Tensor
from tabicl.prior.data_reader import DataReader
from tabicl.train.train_config import build_parser



class UCREvaluator:
    def __init__(self, model: TabICL, reader: DataReader, device: str = "cuda"):
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
                logits = self.model(
                    X,
                    y_train.unsqueeze(0),  # y_train: (B=1, train_size)
                    embed_with_test=False,
                    return_logits=True,
                )

            # 提取测试集的预测结果
            preds = logits.argmax(dim=-1).squeeze(0)  # (test_size,)    # 这种做法过于粗暴了，实际上它的TabICLClassifier是有很多trick 的
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
        "max_classes": config.max_classes,
        "embed_dim": config.embed_dim,
        "col_num_blocks": config.col_num_blocks,
        "col_nhead": config.col_nhead,
        "col_num_inds": config.col_num_inds,
        "row_num_blocks": config.row_num_blocks,
        "row_nhead": config.row_nhead,
        "row_num_cls": config.row_num_cls,
        "row_rope_base": config.row_rope_base,
        "icl_num_blocks": config.icl_num_blocks,
        "icl_nhead": config.icl_nhead,
        "ff_factor": config.ff_factor,
        "dropout": config.dropout,
        "activation": config.activation,
        "norm_first": config.norm_first,
    }

    model = TabICL(**model_config)  # 使用 ** 解包字典参数
    
    # 初始化模型和数据读取器

    reader = DataReader(data_path="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/data/")  # 替换为实际数据读取器初始化代码
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)  # 将模型移动到指定设备
    
    checkpoint = torch.load("/data0/fangjuntao2025/tabicl-main/checkpoints/step-9900.ckpt", map_location=device, weights_only=True)  #"/data0/fangjuntao2025/tabicl-main/src/tabicl/checkpoints/step-23800.ckpt"

    # Load model state
    if "state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain model state")
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Model device: {next(model.parameters()).device}")
    # 初始化评估器
    evaluator = UCREvaluator(model, reader, device)

    # 评估并打印结果
    results = evaluator.evaluate()
    
    evaluator.print_results(results)
