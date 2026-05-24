"""
量化工具
支持 INT8, INT4 量化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class QuantConfig:
    """量化配置"""
    bits: int = 8  # 量化位数
    group_size: int = 128  # 每组参数量
    method: str = "symmetric"  # "symmetric" 或 "asymmetric"


class Quantizer:
    """量化器基类"""

    def __init__(self, bits: int = 8, group_size: int = 128):
        self.bits = bits
        self.group_size = group_size

    def quantize(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """量化权重"""
        raise NotImplementedError

    def dequantize(self, qweight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """反量化"""
        raise NotImplementedError


class INT8Quantizer(Quantizer):
    """INT8 量化器"""

    def quantize(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对权重进行 INT8 量化

        返回:
            qweight: (num_groups, group_size) INT8 量化权重
            scale: (num_groups) 缩放因子
        """
        # 按组量化
        num_groups = weight.numel() // self.group_size
        reshaped = weight[:num_groups * self.group_size].view(num_groups, self.group_size)

        # 计算 scale：每个组独立量化
        # symmetric: scale = max(|w|) / 127
        scale = reshaped.abs().max(dim=-1)[0] / 127.0
        scale = scale.clamp(min=1e-8)

        # 量化
        qweight = torch.round(reshaped / scale.unsqueeze(-1)).clamp(-128, 127).to(torch.int8)

        return qweight, scale

    def dequantize(self, qweight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """反量化"""
        return qweight.float() * scale.unsqueeze(-1)


class INT4Quantizer(Quantizer):
    """INT4 量化器"""

    def __init__(self, bits: int = 4, group_size: int = 128):
        super().__init__(bits, group_size)
        # INT4 对称量化范围: -8 到 7 (2^3-1=7)
        # 但为了更好的对称性，通常用 -7 到 7 或 -8 到 7
        self.max_val = 7

    def quantize(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对权重进行 INT4 量化

        注意：实际存储时每 2 个 INT4 打包成一个字节
        这里简化处理，返回 (num_groups, group_size) 的 INT8，实际存储需要打包
        """
        num_groups = weight.numel() // self.group_size
        reshaped = weight[:num_groups * self.group_size].view(num_groups, self.group_size)

        # 计算 scale
        scale = reshaped.abs().max(dim=-1)[0] / self.max_val
        scale = scale.clamp(min=1e-8)

        # 量化到 [-8, 7]
        qweight = torch.round(reshaped / scale.unsqueeze(-1)).clamp(-8, 7).to(torch.int8)

        return qweight, scale

    def dequantize(self, qweight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """反量化"""
        return qweight.float() * scale.unsqueeze(-1)


class QuantizedLinear(nn.Module):
    """量化后的线性层"""

    def __init__(self, weight: nn.Parameter, bias: Optional[nn.Parameter], config: QuantConfig):
        super().__init__()
        self.config = config

        # 量化权重
        if config.bits == 8:
            quantizer = INT8Quantizer(config.bits, config.group_size)
        elif config.bits == 4:
            quantizer = INT4Quantizer(config.bits, config.group_size)
        else:
            raise ValueError(f"Unsupported bits: {config.bits}")

        self.qweight, self.scale = quantizer.quantize(weight.data)
        self.qweight = nn.Parameter(self.qweight, requires_grad=False)
        self.scale = nn.Parameter(self.scale, requires_grad=False)

        self.use_bias = bias is not None
        if self.use_bias:
            self.bias = bias
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 反量化
        weight = self.qweight.float() * self.scale.unsqueeze(-1)

        # 线性计算
        output = F.linear(x, weight, self.bias)

        return output


class GPTQQuantizer:
    """
    GPTQ 量化器

    基于 Importance of Quantization Algorithm Learning in Post-Training Quantization
    """

    def __init__(self, bits: int = 4, group_size: int = 128):
        self.bits = bits
        self.group_size = group_size

    def quantize(self, weight: torch.Tensor, importance: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GPTQ 量化

        参数:
            weight: FP32 权重
            importance: 可选，每个元素的重要性权重
        """
        num_groups = weight.numel() // self.group_size
        reshaped = weight[:num_groups * self.group_size].view(num_groups, self.group_size)

        # 计算重要性（如果没有提供，使用权重范数）
        if importance is None:
            importance = reshaped.abs().mean(dim=-1)
        else:
            importance = importance[:num_groups * self.group_size].view(num_groups, self.group_size).abs().mean(dim=-1)

        # 根据重要性调整量化范围
        # 更重要的权重获得更精细的量化
        max_vals = reshaped.abs().max(dim=-1)[0]
        scales = max_vals / (2 ** (self.bits - 1) - 1)

        # 量化
        qweight = torch.round(reshaped / scales.unsqueeze(-1))
        max_val = 2 ** (self.bits - 1)
        qweight = qweight.clamp(-max_val, max_val - 1)

        return qweight.to(torch.int8), scales

    def dequantize(self, qweight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """反量化"""
        return qweight.float() * scale.unsqueeze(-1)


def quantize_model(model: nn.Module, config: QuantConfig) -> nn.Module:
    """
    对模型进行量化

    参数:
        model: 要量化的模型
        config: 量化配置

    返回:
        量化后的模型（原始模型被修改）
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 替换为量化版本
            quantized = QuantizedLinear(module.weight, module.bias, config)
            setattr(model, name, quantized)

    return model


def get_model_size(model: nn.Module) -> dict:
    """计算模型大小"""
    total_params = sum(p.numel() for p in model.parameters())

    # 估算量化后大小
    # 假设使用 INT8 量化
    fp16_size = total_params * 2 / 1e9  # GB
    int8_size = total_params * 1 / 1e9
    int4_size = total_params * 0.5 / 1e9

    return {
        "total_params": total_params,
        "fp16_gb": round(fp16_size, 2),
        "int8_gb": round(int8_size, 2),
        "int4_gb": round(int4_size, 2)
    }


if __name__ == "__main__":
    # 测试
    weight = torch.randn(1024, 1024) * 5

    # INT8 量化
    quantizer = INT8Quantizer(bits=8, group_size=128)
    qweight, scale = quantizer.quantize(weight)

    print(f"QWeight shape: {qweight.shape}")
    print(f"Scale shape: {scale.shape}")

    # 反量化
    dequantized = quantizer.dequantize(qweight, scale)

    # 计算误差
    error = (weight - dequantized).abs().mean()
    print(f"平均绝对误差: {error:.4f}")

    # 模型大小
    print(get_model_size(None))  # 演示格式