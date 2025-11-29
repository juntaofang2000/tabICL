import torch
import numpy as np
from pathlib import Path
from tabicl import MantisICLClassifier

import pandas as pd
import os
from tabicl.prior.data_reader import DataReader

class UCREvaluator:
    def __init__(self,                  UEA_data_path: str = "/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UEAData/",
                 UCR_data_path: str = "/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UCRdata/"):
        """
        初始化评估器。

        Parameters
        ----------
        data_path : str
            UCR 数据集的路径。
        """
        self.reader = DataReader(
            UEA_data_path=UEA_data_path,
            UCR_data_path=UCR_data_path,
            transform_ts_size=512
        )
        self.results = []
        # 初始化一次 MantisICLClassifier 并在多个数据集上复用，避免重复加载 checkpoint
        self.clf = MantisICLClassifier(
            verbose=False,
            n_estimators=1,  # 取消数据增强
            checkpoint_version="tabicl-classifier-v1.1-0506.ckpt",
            model_path="/data0/fangjuntao2025/tabicl-main/src/tabicl/checkpointsMantisICL12blocks/step-42300.ckpt",
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

            # 确保数据形状正确
            if len(X_train.shape) == 3:
                X_train = X_train.squeeze(1)
            if len(X_test.shape) == 3:
                X_test = X_test.squeeze(1)

            # 复用已创建的分类器实例，只调用 fit() 来为当前数据集准备上下文
            self.clf.fit(X_train, y_train)

            # 预测并计算准确率
            y_pred = self.clf.predict(X_test)
            accuracy = np.mean(y_pred == y_test)

            return accuracy

        except Exception as e:
            print(f"Failed to evaluate dataset {dataset_name}: {e}")
            return 0.0

    def evaluate_all(self) -> None:
        """
        评估所有 UCR 数据集。
        """
        print("Starting evaluation on UCR datasets...")
        total_accuracy = 0.0
        evaluated_count = 0

        for dataset_name in self.reader.dataset_list_ucr:
            accuracy = self.evaluate_dataset(dataset_name)
            if accuracy > 0:  # 只统计成功评估的数据集
                self.results.append((dataset_name, accuracy))
                total_accuracy += accuracy
                evaluated_count += 1
                print(f"{dataset_name}: {accuracy:.4f}")

        # 计算平均准确率
        avg_accuracy = total_accuracy / evaluated_count if evaluated_count > 0 else 0.0
        
        print("\nEvaluation Results:")
        print("-----------------------------")
        print(f"Total datasets evaluated: {evaluated_count}")
        print(f"Average accuracy: {avg_accuracy:.4f}")
        print("-----------------------------")

        # 保存详细结果
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
        
        # 保存汇总结果
        avg_accuracy = sum(acc for _, acc in self.results) / len(self.results)
        with open(results_dir / "summary.txt", "w") as f:
            f.write(f"Total datasets: {len(self.results)}\n")
            f.write(f"Average accuracy: {avg_accuracy:.4f}\n")


def read_ucr_tsv(train_path, test_path):
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

def batch_test_ucr(ucr_root):
    """
    批量读取 UCR 路径下所有数据集并评测。
    ucr_root: UCRArchive_2018 路径（如 /data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/data/UCRArchive_2018）
    """
    results = []
    # 只初始化一次模型
    clf = MantisICLClassifier(verbose=False, checkpoint_version="tabicl-classifier-v1.1-0506.ckpt", model_path="/data0/fangjuntao2025/tabicl-main/src/tabicl/checkpointsMantisICL12blocksFrozeMantis/step-70000.ckpt")
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

if __name__ == "__main__":
    # 1. 原有 DataReader 方式评估所有 UCR 数据集
    data_path = "/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/data/"
    evaluator = UCREvaluator(  UEA_data_path="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UEAData/",
        UCR_data_path= "/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UCRdata/",)
    evaluator.evaluate_all()

    # 2. 新增：批量读取 UCRArchive_2018 下所有数据集并评测 （不改变原始时序长度）
    # ucr_root = "/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/data/UCRArchive_2018"
    # print("\n批量评测 UCRArchive_2018 下所有数据集：")
    # batch_test_ucr(ucr_root)
