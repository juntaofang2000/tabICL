from __future__ import annotations

from typing import Optional, List
from torch import nn, Tensor

from .embedding import ColEmbedding
from .interaction import RowInteraction
from .learning import ICLearning
from .inference_config import InferenceConfig

from tabicl.model.mantis_dev.architecture.architecture import Mantis8M
import torch

class MantisICL(nn.Module):
    """A Tabular In-Context Learning Foundation Model.

    TabICL is a transformer-based architecture for in-context learning on tabular data to make
    predictions without fine-tuning. It processes tabular data through three sequential stages:

    1. Column-wise embedding creates distribution-aware embeddings
    2. Row-wise interaction captures interactions between features within each row
    3. Dataset-wise in-context learning to learn patterns from labeled examples and make predictions

    For datasets with more than `max_classes` classes, TabICL switches to hierarchical lassification
    to recursively partition classes into subgroups, forming a multi-level classification tree.

    Parameters
    ----------
    max_classes : int, default=10
        Number of classes that the model supports natively. If the number of classes
        in the dataset exceeds this value, hierarchical classification is used.

    embed_dim : int, default=128
        Model dimension used in the column / row embedding transformers. For the in-context
        learning transformer, the dimension is this value multiplied by the number of CLS tokens.


    icl_num_blocks : int, default=12
        Number of transformer blocks in the in-context learning transformer

    icl_nhead : int, default=4
        Number of attention heads in the in-context learning transformer

    ff_factor : int, default=2
        Expansion factor for feedforward networks across all components

    dropout : float, default=0.0
        Dropout probability across all components

    activation : str or unary callable, default="gelu"
        Activation function used throughout the model

    norm_first : bool, default=True
        If True, uses pre-norm architecture across all components

    train_mantis : bool, default=False
        If True, Mantis8M will be trained. If False, pre-trained weights will be loaded.
    """

    def __init__(
        self,
        max_classes: int = 60,
        icl_num_blocks: int = 12,
        icl_nhead: int = 4,
        ff_factor: int = 2,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        icl_dim: int = 512,   #  跟mantis 的 hidden_dim 对应  256 -> 512
        train_mantis: bool =True,
    ):
        super().__init__()
        self.max_classes = max_classes
        self.icl_num_blocks = icl_num_blocks
        self.icl_nhead = icl_nhead
        self.ff_factor = ff_factor
        self.dropout = dropout
        self.activation = activation
        self.norm_first = norm_first
        self.icl_dim = icl_dim
        self.train_mantis = train_mantis
        print(f"train_mantis set to {self.train_mantis}")

        # Initialize Mantis8M

        self.mantis_model = Mantis8M(
            seq_len=512,
            hidden_dim=512,  # 256 -> 512
            num_patches=32,
            scalar_scales=None,
            hidden_dim_scalar_enc=32,
            epsilon_scalar_enc=1.1,
            transf_depth=6,
            transf_num_heads=8,
            transf_mlp_dim=512,
            transf_dim_head=128,
            transf_dropout=0.1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            pre_training=False  # forward 的时候没有 prj 头
        )

        # 无论 train_mantis True/False，都加载预训练权重
        # pretrained_path = "/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/Models/Mantis/Mantis_cheickpoint"
        # self.mantis_model = self.mantis_model.from_pretrained(pretrained_path)
        print("mantis 256 -> 512")
        if not self.train_mantis:
            for param in self.mantis_model.parameters():
                param.requires_grad_(False)
            print("Mantis8M fully frozen ")

        # 否则，只冻结 projection 层
        else:
            if hasattr(self.mantis_model, "prj"):
                for name, param in self.mantis_model.prj.named_parameters():
                    param.requires_grad_(False)
                print("Only Frozen Mantis projection layer (mantis.prj) ")

        # Load pre-trained weights if not training
        # if not self.train_mantis:
        #     # MODEL_PATH = "/data0/fangjuntao2025/CauKer/CauKer-main/Models/Mantis/checkpoint/Graph100-k6P520250915"
        #     # model_params = torch.load(MODEL_PATH, weights_only=True)
        #     # self.mantis_model.load_state_dict(model_params)
        #     self.mantis_model = self.mantis_model.from_pretrained("/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/Models/Mantis/Mantis_cheickpoint")
        #     # for param in self.mantis_model.parameters():
        #     #     param.requires_grad_(False)  # 冻结参数
        #     print("Loaded pre-trained Mantis8M weights and unfroze the model parameters------------------------------------.")

        self.icl_predictor = ICLearning(
            max_classes=max_classes,
            d_model=icl_dim,
            num_blocks=icl_num_blocks,
            nhead=icl_nhead,
            dim_feedforward=icl_dim * ff_factor,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )

    def _train_forward(
        self, X: Tensor, y_train: Tensor, d: Optional[Tensor] = None, embed_with_test: bool = False
    ) -> Tensor:
        """Column-wise embedding -> row-wise interaction -> dataset-wise in-context learning for training.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (B, T, H) where:
             - B is the number of tables      数据集数量    
             - T is the number of samples (rows)  一条时序
             - H is the number of features (columns)  时序长度(embeding 特征数)： 512 
            The first train_size positions contain training samples, and the remaining positions contain test samples.

        y_train : Tensor
            Training labels of shape (B, train_size) where:
             - B is the number of tables
             - train_size is the number of training samples provided for in-context learning

        d : Optional[Tensor], default=None
            The number of features per dataset.

        Returns
        -------
        Tensor
            Raw logits of shape (B, T, max_classes), which will be further handled by the training code.
        """

        B, T, H = X.shape
        train_size = y_train.shape[1]
        assert train_size <= T, "Number of training samples exceeds total samples"

        # Check if d is provided and has the same length as the number of features
        if d is not None and len(d.unique()) == 1 and d[0] == H:
            d = None

        # Split X into T sub-tensors of shape [B, 1, H]
        # X_split = torch.split(X, 1, dim=1)  # List of [B, 1, H] tensors

        # # Process each sub-tensor through Mantis8M and collect representations
        # representations = []
        # for x in X_split:
        #     # x shape: [B, 1, H]
        #     rep = self.mantis_model(x)  # rep shape: [1, D]
        #     rep = rep.unsqueeze(1)     # Reshape to [B, 1, D]
        #     representations.append(rep)

        # # Concatenate all representations along the sequence dimension
        # representations = torch.cat(representations, dim=1)  # shape: [B, T, D]
        X_reshaped = X.reshape(-1, 1, H)  # shape: [B*T, 1, H]
        representations = self.mantis_model(X_reshaped)  # rep shape: [B*T, D]
        representations = representations.reshape(B, T, -1)  # Reshape back to [B, T, D]

        # Dataset-wise in-context learning
        out = self.icl_predictor(representations, y_train=y_train)
        # for name, param in self.named_parameters():
        #     if not param.requires_grad:
        #         print(f"Parameter {name} is frozen and unused.")
                
        # print(f"Mantis8M parameters used: {self.mantis_model.training}")
        # print(f"ICLearning parameters used: {self.icl_predictor.training}")

        return out

    def _inference_forward(
        self,
        X: Tensor,
        y_train: Tensor,
        feature_shuffles: Optional[List[List[int]]] = None,
        embed_with_test: bool = False,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        inference_config: InferenceConfig = None,
    ) -> Tensor:
        """Column-wise embedding -> row-wise interaction -> dataset-wise in-context learning.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (B, T, H) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features (columns)
            The first train_size positions contain training samples, and the remaining positions contain test samples.

        y_train : Tensor
            Training labels of shape (B, train_size) where:
             - B is the number of tables
             - train_size is the number of training samples provided for in-context learning

        feature_shuffles : Optional[List[List[int]]], default=None
            A list of feature shuffle patterns for each table in the batch.
            When provided, indicates that X contains the same table with different feature orders.
            In this case, column-wise embeddings are computed once and then shuffled accordingly.

        embed_with_test : bool, default=False
            If True, allow training samples to attend to test samples during embedding

        return_logits : bool, default=True
            If True, return raw logits instead of probabilities

        softmax_temperature : float, default=0.9
            Temperature for the softmax function

        inference_config: InferenceConfig
            Inferenece configuration

        Returns
        -------
        Tensor
            Raw logits or probabilities for test samples of shape (B, test_size, num_classes)
            where test_size = T - train_size
        """

        train_size = y_train.shape[1]
        assert train_size <= X.shape[1], "Number of training samples exceeds total samples"

        if inference_config is None:
            inference_config = InferenceConfig()

        # # Split X into T sub-tensors of shape [B, 1, H]
        # X_split = torch.split(X, 1, dim=1)  # List of [B, 1, H] tensors  # 

        # self.mantis_model.pre_training  =  False
        # # Process each sub-tensor through Mantis8M and collect representations
        # representations = []
        # for x in X_split:
        #     # x shape: [B, 1, H]
        #     rep = self.mantis_model(x)  
        #     rep = rep.unsqueeze(1)  # rep shape: [B, 1, D] 
        #     representations.append(rep)

        # # Concatenate all representations along the sequence dimension
        # representations = torch.cat(representations, dim=1)  # shape: [B, T, D]
        # Reshape X to [B*T, 1, H] and process all at once
        B, T, H = X.shape
        X_reshaped = X.reshape(-1, 1, H)  # shape: [B*T, 1, H]
        self.mantis_model.pre_training = False
        # representations = self.mantis_model(X_reshaped)  # rep shape: [B*T, D]
        # representations = representations.view(B, T, -1)  # Reshape back to [B, T, D]
        
        
        # 修改后
        batch_size = 32  # 可根据GPU内存调整
        BT = X_reshaped.shape[0]
        representations_list = []

        for i in range(0, BT, batch_size):
            end_idx = min(i + batch_size, BT)
            batch_X = X_reshaped[i:end_idx]  # [batch_size, 1, H]
            batch_rep = self.mantis_model(batch_X)  # [batch_size, D]
            representations_list.append(batch_rep)

        representations = torch.cat(representations_list, dim=0)  # [B*T, D]
        representations = representations.reshape(B, T, -1)          
        # Dataset-wise in-context learning
        out = self.icl_predictor(
            representations,
            y_train=y_train,
            return_logits=return_logits,
            softmax_temperature=softmax_temperature,
            mgr_config=inference_config.ICL_CONFIG,
        )

        return out

    def forward(
        self,
        X: Tensor,
        y_train: Tensor,
        d: Optional[Tensor] = None,
        feature_shuffles: Optional[List[List[int]]] = None,
        embed_with_test: bool = False,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        inference_config: InferenceConfig = None,
    ) -> Tensor:
        """Column-wise embedding -> row-wise interaction -> dataset-wise in-context learning.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (B, T, H) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features (columns)
            The first train_size positions contain training samples, and the remaining positions contain test samples.

        y_train : Tensor
            Training labels of shape (B, train_size) where:
             - B is the number of tables
             - train_size is the number of training samples provided for in-context learning

        d : Optional[Tensor], default=None
            The number of features per dataset. Used only in training mode.

        feature_shuffles : Optional[List[List[int]]], default=None
            A list of feature shuffle patterns for each table in the batch. Used only in training mode.
            When provided, indicates that X contains the same table with different feature orders.
            In this case, column-wise embeddings are computed once and then shuffled accordingly.

        embed_with_test : bool, default=False
            If True, allow training samples to attend to test samples during embedding

        return_logits : bool, default=True
            If True, return raw logits instead of probabilities. Used only in training mode.

        softmax_temperature : float, default=0.9
            Temperature for the softmax function. Used only in training mode.

        inference_config: InferenceConfig
            Inferenece configuration. Used only in training mode.

        Returns
        -------
        Tensor
            For training mode:
              Raw logits of shape (B, T-train_size, max_classes), which will be further handled by the training code.

            For inference mode:
              Raw logits or probabilities for test samples of shape (B, T-train_size, num_classes).
        """

        if self.training:
            out = self._train_forward(X, y_train, d=d, embed_with_test=embed_with_test)
        else:
            out = self._inference_forward(
                X,
                y_train,
                feature_shuffles=feature_shuffles,
                embed_with_test=embed_with_test,
                return_logits=return_logits,
                softmax_temperature=softmax_temperature,
                inference_config=inference_config,
            )

        return out
