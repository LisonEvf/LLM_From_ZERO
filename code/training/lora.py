"""
LoRA (Low-Rank Adaptation) 实现
用于高效微调大模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class LoRAConfig:
    """LoRA 配置"""
    r: int = 4  # 低秩维度
    lora_alpha: int = 8  # 缩放因子
    target_modules: List[str] = None  # 目标模块名列表
    lora_dropout: float = 0.05
    bias: str = "none"  # "none", "lora_only", "all"


class LoRALinear(nn.Module):
    """LoRA 适配的线性层"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: int = 8,
        dropout: float = 0.05,
        bias: bool = True
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 原始权重（冻结）
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features),
            requires_grad=False
        )
        self.use_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.register_parameter('bias', None)

        # LoRA 参数（A 和 B）
        # A: (rank, in_features), 初始化为小型随机值
        # B: (out_features, rank), 初始化为零
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始输出
        output = F.linear(x, self.weight, self.bias)

        # LoRA 更新: W = W₀ + (alpha/rank) * B @ A
        lora_update = (self.lora_B @ self.lora_A) * self.scaling

        # 应用 dropout（仅在训练时）
        if self.training:
            x = self.dropout(x)

        output = output + x @ lora_update.t()

        return output

    def merge_weights(self):
        """将 LoRA 权重合并到原始权重（用于推理）"""
        lora_weight = (self.lora_B @ self.lora_A) * self.scaling
        self.weight.data = self.weight.data + lora_weight.t()
        # 标记 LoRA 参数为不需要梯度
        self.lora_A.requires_grad = False
        self.lora_B.requires_grad = False


class LoRAModel(nn.Module):
    """应用 LoRA 的模型"""

    def __init__(
        self,
        base_model: nn.Module,
        config: LoRAConfig
    ):
        super().__init__()
        self.base_model = base_model
        self.config = config

        # 需要替换的层
        self.replace_layers = {}

        # 遍历模型的所有模块
        for name, module in base_model.named_modules():
            if self._should_replace(name, module):
                # 创建 LoRA 版本
                lora_layer = self._create_lora_layer(module, name)
                self._replace_module(name, lora_layer)

        print(f"LoRA applied to {len(self.replace_layers)} modules")

    def _should_replace(self, name: str, module: nn.Module) -> bool:
        """判断是否应该替换该层"""
        if self.config.target_modules is None:
            return isinstance(module, nn.Linear)

        return any(target in name for target in self.config.target_modules)

    def _create_lora_layer(self, module: nn.Linear, name: str) -> LoRALinear:
        """创建 LoRA 版本的层"""
        return LoRALinear(
            in_features=module.in_features,
            out_features=module.out_features,
            rank=self.config.r,
            alpha=self.config.lora_alpha,
            dropout=self.config.lora_dropout,
            bias=module.bias is not None
        )

    def _replace_module(self, name: str, lora_layer: LoRALinear):
        """替换模型中的层"""
        # 获取父模块和属性名
        parts = name.rsplit('.', 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = self.base_model.get_submodule(parent_name)
        else:
            parent = self.base_model
            attr_name = name

        # 复制原始权重
        original = parent.get_submodule(attr_name) if '.' in name else getattr(parent, attr_name)
        lora_layer.weight.data = original.weight.data.clone()
        if original.bias is not None:
            lora_layer.bias.data = original.bias.data.clone()

        # 替换
        setattr(parent, attr_name, lora_layer)
        self.replace_layers[name] = lora_layer

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def merge_weights(self):
        """合并所有 LoRA 权重到原始层"""
        for name, lora_layer in self.replace_layers.items():
            lora_layer.merge_weights()

    def get_trainable_params(self):
        """获取可训练参数"""
        trainable = []
        for name, lora_layer in self.replace_layers.items():
            trainable.extend([lora_layer.lora_A, lora_layer.lora_B])
        return trainable


def apply_lora_to_model(model: nn.Module, config: LoRAConfig) -> LoRAModel:
    """将 LoRA 应用到模型"""
    return LoRAModel(model, config)


def count_lora_params(model: LoRAModel) -> dict:
    """统计 LoRA 参数数量"""
    total_params = sum(p.numel() for p in model.base_model.parameters())

    trainable_params = 0
    for name, lora_layer in model.replace_layers.items():
        trainable_params += lora_layer.lora_A.numel() + lora_layer.lora_B.numel()

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_params / total_params
    }


if __name__ == "__main__":
    # 测试
    from .gpt import create_gpt_config, GPTModel

    # 创建基础模型
    config = create_gpt_config("gpt2-small")
    model = GPTModel(**config)

    # 应用 LoRA
    lora_config = LoRAConfig(
        r=4,
        lora_alpha=8,
        target_modules=['W_q', 'W_k', 'W_v', 'W_o'],  # 匹配 attention 层
        lora_dropout=0.05
    )

    lora_model = apply_lora_to_model(model, lora_config)

    # 统计
    stats = count_lora_params(lora_model)
    print(f"总参数: {stats['total_params'] / 1e6:.2f}M")
    print(f"可训练参数: {stats['trainable_params'] / 1e3:.2f}K")
    print(f"可训练比例: {stats['trainable_ratio'] * 100:.2f}%")

    # 测试
    x = torch.randint(0, config["vocab_size"], (2, 10))
    logits = lora_model(x)
    print(f"Output shape: {logits.shape}")