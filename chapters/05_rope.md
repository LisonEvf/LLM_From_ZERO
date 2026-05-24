# 第五章：让上下文更长 — RoPE旋转位置编码

## 故事开场：外推的困境

Alice用GPT-2写了一个很棒的故事生成器。她测试了一个句子：

```
Prompt: "在很久很久以前，有一只猫坐在"
Model: "椅子上看书" ✓

Prompt: "在很久很久以前，有一只猫坐在椅子上看书，突然"
Model: "它发现" ✓
```

看起来不错！但当她输入一个超长上下文时，问题出现了：

```
Prompt: [10000个token的超长上下文...]
Model的预测开始混乱，甚至重复自己的输出
```

问题根源：**模型无法处理超过训练长度的位置**。

## 5.1 位置编码的问题

回顾第二章，原始Transformer使用 **Sinusoidal位置编码**：

```
位置m的编码: PE(m, i) = sin(m / 10000^(2i/d))

其中 i = 0, 1, 2, ..., d/2
```

### 问题1：位置值太大

当 m（位置）非常大时，sin(m / 10000^(2i/d)) 可能无法区分：

```
PE(1000000, 0) = sin(1000000 / 1) = sin(1000000) ≈ sin(1000000 mod 2π) ≈ sin(5.28) ≈ -0.88
PE(1000001, 0) = sin(1000001) ≈ sin(6.44) ≈ 0.90  (相邻位置差异明显)

但对于高维度 i:
PE(1000000, 256) = sin(1000000 / 10000^(512/512)) = sin(1000000 / 10000) = sin(100) ≈ -0.51
PE(1000001, 256) = sin(1000001 / 10000) ≈ sin(100.0001) ≈ -0.51 (几乎相同！)
```

### 问题2：可学习编码无法外推

GPT-2使用可学习的位置嵌入表：

```python
self.position_embedding = nn.Embedding(max_seq_len, d_model)
# max_seq_len = 1024 (训练时)
# 但推理时可能需要 2048, 4096... 无法泛化
```

当位置超过训练长度时，模型没有"见过"这个位置，嵌入是随机的。

## 5.2 RoPE的核心思想

RoPE（Rotary Position Encoding，旋转位置编码）来自苏剑林的论文 *RoFormer*。

核心洞察：**用旋转矩阵对位置信息进行编码**

数学表达：

```
对于位置m的Query向量 q_m 和 Key向量 k_m：

q_m' = R(m, θ) · q_m
k_m' = R(m, θ) · k_m

其中 R(m, θ) 是旋转矩阵：

R(m, θ) = ┌ cos(mθ)  -sin(mθ) ┐
          │ sin(mθ)   cos(mθ) │
          └                   ┘
```

### 关键性质：相对位置编码

```
q_m' · k_n' = (R(m,θ)q) · (R(n,θ)k)
            = q · k  (当 m-n = const 时) ← 只依赖相对位置！
```

这意味着：**旋转后的内积只依赖于 (m-n)**，即相对位置，而不是绝对位置。

## 5.3 RoPE的数学直觉

为什么旋转有效？让我们看一个例子：

```python
import numpy as np

def rotate_vector(vec, angle):
    """对2维向量进行旋转"""
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    # 旋转矩阵
    R = np.array([[cos_a, -sin_a],
                  [sin_a,  cos_a]])
    return R @ vec

def rope_score(q, k, m, n, theta_base=10000):
    """
    计算RoPE下q和k的注意力分数
    只依赖于 (m-n)
    """
    # 旋转
    angle = (m - n) / theta_base
    q_rot = rotate_vector(q, angle * m)
    k_rot = rotate_vector(k, angle * n)

    # 内积
    return np.dot(q_rot, k_rot)

# 示例
q = np.array([1.0, 0.0])  # 沿x轴
k = np.array([1.0, 0.0])
theta_base = 10000

# 位置0和位置0
score_0_0 = rope_score(q, k, 0, 0, theta_base)

# 位置5和位置3（相对位置=2）
score_5_3 = rope_score(q, k, 5, 3, theta_base)

# 位置100和位置98（相对位置=2）
score_100_98 = rope_score(q, k, 100, 98, theta_base)

print(f"score(0,0): {score_0_0:.4f}")
print(f"score(5,3): {score_5_3:.4f}")  # 应该等于 score(100,98)
print(f"score(100,98): {score_100_98:.4f}")
```

输出表明：不同的绝对位置，相同的相对位置 → 相同的注意力分数。

## 5.4 RoPE在多头注意力中的应用

RoPE不作用于整个向量，而是作用于向量的**不同维度对**：

```python
# code/model/rope.py
"""
RoPE (Rotary Position Encoding) 实现
"""

import torch
import torch.nn as nn
import math


def precompute_freqs_cis(dim, end, theta=10000.0):
    """
    预计算旋转角度

    参数:
        dim: 维度（通常是 d_model / num_heads）
        end: 最大位置
        theta: 基础角度
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)
    freqs = torch.polar(torch.ones_like(freqs), freqs)  # 转为复数形式
    return freqs


def apply_rotary_emb(q, k, freqs):
    """
    应用旋转位置编码

    参数:
        q: (batch, num_heads, seq_len, head_dim)
        k: (batch, num_heads, seq_len, head_dim)
        freqs: (seq_len, head_dim/2) 复数
    """
    # 分离实部和虚部，应用旋转
    # 旋转公式: (x + yi) * (cos + isin) = x*cos - y*sin + i(x*sin + y*cos)

    # q: (batch, num_heads, seq_len, head_dim)
    # freqs: (seq_len, head_dim/2) -> 需要unsqueeze扩展
    freqs = freqs.unsqueeze(0).unsqueeze(0)  # -> (1, 1, seq_len, head_dim/2)

    # 重组为复数格式
    q_complex = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_complex = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))

    # 旋转
    q_rot = torch.view_as_real(q_complex * freqs)
    k_rot = torch.view_as_real(k_complex * freqs)

    return q_rot.flatten(-2), k_rot.flatten(-2)


# 示例
batch_size = 2
num_heads = 8
seq_len = 10
head_dim = 64

q = torch.randn(batch_size, num_heads, seq_len, head_dim)
k = torch.randn(batch_size, num_heads, seq_len, head_dim)

# 预计算频率
freqs = precompute_freqs_cis(head_dim, seq_len)

# 应用RoPE
q_rot, k_rot = apply_rotary_emb(q, k, freqs)

print(f"Q after RoPE: {q_rot.shape}")
print(f"K after RoPE: {k_rot.shape}")
```

## 5.5 Llama中的RoPE配置

LLaMA使用RoPE，并引入了 **ALibi（Attention with Linear Biases）** 的思想做了一定调整：

| 模型 | 上下文长度 | RoPE θ |
|------|------------|--------|
| LLaMA-7B | 2048 | 10000 |
| LLaMA-13B | 2048 | 10000 |
| LLaMA-33B | 2048 | 10000 |
| LLaMA-65B | 2048 | 10000 |
| LLaMA2-70B | 4096 | 10000 |

### 扩展上下文：YaRN

当需要更长上下文时，可以使用 **YaRN（Yet another RoPE extensioN）**：

1. **缩放**：对频率进行缩放
2. **衰减**：对注意力分数应用位置衰减

```python
def apply_rope_with_scale(q, k, position_ids, scale=1.0, original_seq_len=2048):
    """
    带缩放的RoPE，用于上下文扩展
    """
    # 频率缩放
    freqs = precompute_freqs_cis(head_dim, max_pos)
    freqs = freqs / scale  # 频率缩小 = 周期增大

    # 应用
    return apply_rotary_emb(q, k, freqs)
```

## 5.6 RoPE vs 其他位置编码

| 编码方式 | 绝对/相对 | 可外推 | 计算效率 | 典型应用 |
|----------|------------|--------|----------|----------|
| Sinusoidal | 绝对 | 可 | 高 | 原始Transformer |
| 可学习 | 绝对 | 不可 | 高 | GPT-2 |
| ALiBi | 相对 | 可 | 高 | FlashAttention |
| RoPE | 相对 | 可 | 中 | LLaMA, GLM |
| RoPE + YaRN | 相对 | 可(长) | 中 | Vicuna, CodeLLaMA |

## 5.7 完整示例：从零实现RoPE GPT

```python
# code/model/rope_gpt.py
"""
使用RoPE的GPT模型
"""

import torch
import torch.nn as nn
import math


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # 预计算频率
        self.freqs = self._precompute_freqs(dim, max_seq_len)

    def _precompute_freqs(self, dim, seq_len):
        freqs = 1.0 / (self.theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(seq_len)
        freqs = torch.outer(t, freqs)
        return torch.polar(torch.ones_like(freqs), freqs)

    def forward(self, seq_len):
        return self.freqs[:seq_len]


def apply_rotary_pos_emb(q, k, freqs):
    """应用旋转位置编码"""
    q_complex = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_complex = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))

    q_rot = torch.view_as_real(q_complex * freqs.unsqueeze(0))
    k_rot = torch.view_as_real(k_complex * freqs.unsqueeze(0))

    return q_rot.flatten(-2), k_rot.flatten(-2)


class RoPEGPTBlock(nn.Module):
    """使用RoPE的GPT Block"""
    def __init__(self, d_model, num_heads, d_ff, max_seq_len):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),  # LLaMA使用SiLU
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.rope = RotaryEmbedding(d_model // num_heads, max_seq_len)

    def forward(self, x):
        seq_len = x.size(1)
        freqs = self.rope(seq_len)

        # 分离QKV
        q, k, v = self.attention.q_proj(x), self.attention.k_proj(x), self.attention.v_proj(x)

        # 应用RoPE
        q, k = apply_rotary_pos_emb(q, k, freqs)

        # Attention
        attn_out, _ = self.attention(q, k, v)
        x = self.norm1(x + attn_out)

        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x
```

## 5.8 本章小结

1. **位置编码问题**：原始编码无法外推到训练长度之外
2. **RoPE核心**：用旋转矩阵编码位置，内积只依赖相对位置
3. **外推能力**：RoPE天然支持任意长度的位置
4. **效率**：预计算频率，复数乘法
5. **应用**：LLaMA、GLM、DeepSeek等现代模型都使用RoPE

### 思考题

1. RoPE为什么只旋转偶数维度？奇数维度保留有什么作用？
2. 如果把θ从10000改为1000，会发生什么？
3. 为什么LLaMA2的上下文长度可以扩展到4096，而LLaMA1只有2048？

### 延伸阅读

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071)