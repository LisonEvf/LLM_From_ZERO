# 第七章：让大模型变小 — 量化技术

## 故事开场：350GB的困境

Bob兴奋地准备用GPT-3-175B搭建一个问答服务。但当他看到硬件要求时，心凉了半截：

```
GPT-3-175B 参数:
- FP16 (16位浮点): 350GB 显存
- 需要: 8×A100 (80GB) = 640GB 总计

典型的消费级GPU: 24GB (RTX 4090)
差距: 640 / 24 ≈ 27倍
```

Alice笑着说："这就是量化的用武之地——**用更少的bit表示权重**。"

## 7.1 量化概述

### 什么是量化？

**量化（Quantization）**：将高精度的数值映射到低精度的表示。

```
FP32 (32位浮点):  1.2345678901234567890 (约32位有效数字)
FP16 (16位浮点):  1.2345 (约5位有效数字)
INT8 (8位整数):   1.2 (约2位有效数字)
INT4 (4位整数):   1  (约0.5位有效数字)
```

### 为什么有效？

深度学习模型对噪声有很强的鲁棒性。微小的权重变化对最终预测影响很小。

```
原始权重: 0.123456789
量化后:   0.12 (INT8, step=0.01)

误差: 0.003456789 (约2.8%)
但模型性能几乎不受影响
```

### 量化挑战

```
问题: 如何选择"量化网格"？

FP32: 连续值，每个值都精确
量化: 离散值，只能表示有限个点

解决方案: 对数-LAW (Logarithmic Asymmetric Quantization)
- 非均匀量化，匹配权重分布
```

## 7.2 量化方法对比

| 方法 | 原理 | 精度损失 | 速度 | 显存节省 |
|------|------|----------|------|----------|
| INT8 | 逐层量化 | 中等 | 2-4× | 50% |
| GPTQ | 逐层量化，利用Hessian | 低 | 2-4× | 75% |
| AWQ | 激活值加权，保留关键权重 | 低 | 2-4× | 75% |
| INT4 | 极端压缩 | 高 | 4-8× | 87.5% |
| NF4 | 4位NormalFloat | 中 | 4-8× | 87.5% |

## 7.3 INT8 量化

最基本的量化方法：**均匀量化**。

```python
# code/inference/quantize.py
"""
INT8 量化实现
"""

import torch
import numpy as np


def quantize_int8(weights, per_channel=False):
    """
    将FP32权重量化到INT8

    参数:
        weights: FP32 权重 (torch.Tensor)
        per_channel: 是否按通道量化

    返回:
        qweights: INT8 量化权重
        scale: 量化缩放因子
    """
    if per_channel:
        # 按通道计算scale
        dim = 0  # 沿着第一个维度（output channel）
        scales = weights.abs().max(dim=dim)[0] / 127.0
        scales = scales.view(-1, 1)  # (out_channels, 1)

        # 量化
        qweights = torch.round(weights / scales).clamp(-128, 127).to(torch.int8)
    else:
        # 全局scale
        scale = weights.abs().max() / 127.0
        qweights = torch.round(weights / scale).clamp(-128, 127).to(torch.int8)

    return qweights, scale


def dequantize_int8(qweights, scale):
    """INT8反量化回FP32"""
    return qweights.float() * scale


# 示例
weights = torch.randn(1024, 1024) * 2.5  # 模拟真实权重分布

qweights, scale = quantize_int8(weights, per_channel=True)
reconstructed = dequantize_int8(qweights, scale)

# 计算误差
error = (weights - reconstructed).abs().mean()
original_std = weights.std()
print(f"量化误差: {error:.4f}")
print(f"误差/标准差比: {error/original_std:.4%}")
```

## 7.4 GPTQ: 基于Hessian的量化

GPTQ（Generative Post-Training Quantization）的核心思想：

> **不是所有权重都同等重要**。用Hessian信息识别关键权重，给予更高精度。

### 算法步骤

```python
# 简化版GPTQ
def gptq_quantize(weights, bit_width=4):
    """
    GPTQ 量化流程

    1. 先对权重做一次SVD-like分解
    2. 识别"重要"权重（在Hessian对角线上权重大的）
    3. 保留这些权重的更高精度
    4. 其余权重量化到目标bit
    """
    # 计算Hessian近似（对角线）
    # H ≈ J^T J, J是梯度
    H_diag = (weights ** 2).mean(axis=-1)

    # 找出重要权重（对角线值大的）
    importance = H_diag / H_diag.sum()

    # 混合精度量化
    # 重要: 更高精度 (如INT8)
    # 不重要: 更低精度 (如INT4)
    threshold = np.percentile(importance, 75)  # top 25%更精确

    high_precision_mask = importance > threshold
    low_precision_mask = ~high_precision_mask

    # 分别量化
    qweights = torch.zeros_like(weights)
    qweights[high_precision_mask] = quantize_high(weights[high_precision_mask], 8)
    qweights[low_precision_mask] = quantize_high(weights[low_precision_mask], 4)

    return qweights
```

## 7.5 AWQ: 激活值加权

AWQ（Activation-aware Weight Quantization）的核心洞察：

> **权重对最终结果的影响，取决于它所乘的激活值**。

"大激活 × 大权重 = 关键权重"

```python
def awq_quantize(weights, activations, bit_width=4):
    """
    AWQ 量化

    1. 计算激活的统计量（每个channel的平均激活）
    2. 权重 × 激活 的比例决定量化的"容差"
    3. 大比例 → 高精度，小比例 → 低精度
    """
    # 激活比例
    act_scales = activations.abs().mean(dim=0).view(-1, 1)

    # 结合权重和激活的信息
    combined_scales = (weights.abs().mean(dim=-1, keepdim=True) * act_scales).clamp(min=1e-8)

    # 按combined_scales分配bit精度
    # 大的 -> 用更多bit
    # 小的 -> 用更少bit

    qweights = quantize_with_scales(weights, combined_scales, bit_width)
    return qweights
```

## 7.6 量化实践

### 使用 transformers 加载量化模型

```python
from transformers import AutoModelForCausalLM, AutoConfig

# GPTQ 量化
# 需要: pip install auto-gptq

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=True  # 激活顺序量化，更准确但更慢
)

# 量化模型
model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    quantize_config=quantize_config
)

# 保存量化模型
model.save_quantized("llama-2-7b-gptq-4bit")
```

### 使用 llama.cpp 进行INT4推理

```bash
# 1. 安装llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake ..
make -j4

# 2. 量化模型
python llama.cpp/convert.py meta-llama/Llama-2-7b/ --outfile llama-2-7b-f16.gguf
./quantize llama-2-7b-f16.gguf llama-2-7b-q4.bin q4_0

# 3. 推理
./main -m llama-2-7b-q4.bin -n 128 -p "The future of AI is"
```

### 量化前后对比

```python
# 量化效果对比
import torch

def compare_quantization(model_name):
    # 原始模型 (FP16)
    model_fp16 = load_model(model_name, dtype=torch.float16)

    # 量化模型 (INT4)
    model_int4 = load_quantized_model(model_name, bits=4)

    # 计算参数量
    params_fp16 = sum(p.numel() for p in model_fp16.parameters())
    params_int4 = sum(p.numel() for p in model_int4.parameters())  # INT4实际存储更少

    # 估算显存
    memory_fp16 = params_fp16 * 2 / 1e9  # GB
    memory_int4 = params_int4 * 0.5 / 1e9  # INT4≈0.5字节

    print(f"FP16 显存: {memory_fp16:.2f} GB")
    print(f"INT4 显存: {memory_int4:.2f} GB")
    print(f"压缩比: {memory_fp16/memory_int4:.1f}×")

# 示例
compare_quantization("meta-llama/Llama-2-7b")
# FP16 显存: 13.5 GB
# INT4 显存: 3.9 GB
# 压缩比: 3.5×
```

## 7.7 量化质量评估

### 困惑度对比

```python
def evaluate_quantization(model_fp16, model_int4, test_data):
    """评估量化模型的质量"""
    # 计算困惑度
    ppl_fp16 = calculate_perplexity(model_fp16, test_data)
    ppl_int4 = calculate_perplexity(model_int4, test_data)

    print(f"FP16 困惑度: {ppl_fp16:.4f}")
    print(f"INT4 困惑度: {ppl_int4:.4f}")
    print(f"损失增加: {(ppl_int4 - ppl_fp16)/ppl_fp16 * 100:.2f}%")

    return ppl_fp16, ppl_int4
```

### 常见基准

| 模型 | 量化 | 困惑度 | 精度损失 |
|------|------|--------|----------|
| LLaMA-7B | FP16 | 5.58 | - |
| LLaMA-7B | GPTQ-INT4 | 5.68 | +1.8% |
| LLaMA-7B | AWQ-INT4 | 5.62 | +0.7% |

## 7.8 本章小结

1. **量化目的**：用更少的bit表示权重，降低显存和加速推理
2. **INT8**：基础量化方法，精度损失可接受
3. **GPTQ**：利用Hessian信息，保留关键权重更高精度
4. **AWQ**：激活值加权，匹配实际计算分布
5. **INT4**：高压缩，但精度损失明显，需结合其他技术

### 思考题

1. 为什么量化主要关注权重而不关注激活值？
2. 为什么某些层的权重对量化更敏感？如何识别它们？
3. 量化后模型能否在任意精度下工作，还是只能在离散精度下工作？

### 延伸阅读

- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
- [llama.cpp: LLM inference in pure C/C++](https://github.com/ggerganov/llama.cpp)