# 第三章：Attention革命 — Transformer架构的诞生

## 故事开场：RNN的困境

Alice学完了词嵌入，兴冲冲地开始处理长文本："The cat which wore a hat and jumped over the fence sat on the chair."

她把这个句子交给 RNN 处理。模型需要回答："What sat on the chair?"

RNN 的问题在于：**它必须从左到右逐步处理**，而且信息在传递过程中会逐渐稀释。

```
Token:  The → cat → which → wore → a → hat → and → jumped → over → the → fence → sat → on → the → chair
         ↓    ↓     ↓       ↓     ↓    ↓     ↓     ↓       ↓    ↓     ↓    ↓    ↓     ↓     ↓
隐藏层:  H0   H1    H2      H3    H4   H5    H6    H7      H8   H9   H10  H11  H12   H13   H14
                                                            ↑
                                                            │
                                                    需要理解"The cat sat on the chair"
                                                    但所有信息都经过14步传递
```

当RNN处理到"chair"时，关于"cat"的信息已经非常微弱了。这就是**长距离依赖问题**（Long-range Dependency Problem）。

**Attention is All You Need** —— 2017年，Google的一篇论文彻底改变了这个局面。

## 3.1 注意力机制的核心思想

Attention的灵感来自人类的注意力：**当我们阅读一句话时，会自动关注重要的词**。

对于句子 "The cat sat on the chair"：
- 问：什么坐在椅子上？
- 答：**猫** (cat)
- 注意力模式：`cat → sat → chair` （猫和椅子是关键词）

Attention的工作原理：

```
输入: [The, cat, sat, on, the, chair]
         ↓
    ┌───────────────────────────────┐
    │  Query: "什么坐在椅子上？"     │
    │  Key: 每个词的"身份标识"       │
    │  Value: 每个词的"语义内容"     │
    └───────────────────────────────┘
         ↓
    Attention计算：
    - Query与所有Key做相似度计算
    - 得到权重（注意力分数）
    - 用权重对Value加权求和
         ↓
    输出: 聚合后的上下文表示
```

### Query、Key、Value 三角关系

```
        Query (查询)
           ↓
           │
    ┌──────┴───────┐
    ↓              ↓
   Key           Key
    ↓              ↓
   Value         Value

Query找相关Key，
用相关Key的Value重建输出
```

类比：图书馆搜索
- **Query**：你想找的书（"机器学习的书"）
- **Key**：每本书的标签/索引
- **Value**：书的实际内容

Attention计算的就是Query和Key的匹配程度。

## 3.2 缩放点积注意力（Scaled Dot-Product Attention）

这是Transformer的核心公式。

```
           Q (Query)                    K (Key)
         (seq_len, d_k)              (seq_len, d_k)
              ↓                           ↓
         [q₀ q₁ ... qₙ]               [k₀ k₁ ... kₙ]
              ↓                           ↓
              └──────────┬───────────────┘
                         ↓
                  QKᵀ / √d_k
                    (相似度矩阵)
                         ↓
                    Softmax
                    (归一化)
                         ↓
                   Attention权重
                    (seq_len, seq_len)
                         ↓
                   乘以 V (Value)
                   (seq_len, d_v)
                         ↓
                    Output
                   (seq_len, d_v)
```

数学表达式：

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) V

其中：
- d_k: Key向量的维度
- √d_k: 缩放因子，防止点积过大导致梯度消失
```

### 从零实现

```python
import torch
import torch.nn.functional as F
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    缩放点积注意力

    参数:
        Q: (batch, seq_len_q, d_k)
        K: (batch, seq_len_k, d_k)
        V: (batch, seq_len_v, d_v)  通常 seq_len_v = seq_len_k
        mask: 可选，注意力掩码

    返回:
        output: (batch, seq_len_q, d_v)
        attention_weights: (batch, seq_len_q, seq_len_k)
    """
    d_k = Q.shape[-1]

    # 1. 计算点积相似度
    scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
    #  scores: (batch, seq_len_q, seq_len_k)

    # 2. 应用掩码（如果提供）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 3. Softmax归一化
    attention_weights = F.softmax(scores, dim=-1)
    #  attention_weights: (batch, seq_len_q, seq_len_k)

    # 4. 加权求和
    output = torch.matmul(attention_weights, V)
    #  output: (batch, seq_len_q, d_v)

    return output, attention_weights


# 示例
batch_size = 2
seq_len = 5
d_k = 64

Q = torch.randn(batch_size, seq_len, d_k)
K = torch.randn(batch_size, seq_len, d_k)
V = torch.randn(batch_size, seq_len, d_k)

output, attn_weights = scaled_dot_product_attention(Q, K, V)

print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attn_weights.shape}")
print(f"注意力权重之和（每行）: {attn_weights[0].sum(dim=-1)}")
```

## 3.3 多头注意力（Multi-Head Attention）

单个注意力头只能关注一种模式。多头注意力让模型同时关注多个方面：

```
输入X:
  ↓
┌────────────────────────────────────────┐
│  Split into H heads                    │
│  X₀  X₁  ...  Xₕ                       │
│   ↓   ↓        ↓                       │
│  Wq₀ Wq₁ ... Wqₕ                       │
│  Wk₀ Wk₁ ... Wkₕ                       │
│  Wv₀ Wv₁ ... Wvₕ                       │
└────────────────────────────────────────┘
        ↓            ↓
   每个头独立做   每个头独立做
   Attention     Attention
        ↓            ↓
   head₀  head₁ ... headₕ
        ↓            ↓
┌────────────────────────────────────────┐
│  Concat + Wᴼ                           │
│  [head₀ | head₁ | ... | headₕ] → Output│
└────────────────────────────────────────┘
```

数学表达式：

```
MultiHead(Q, K, V) = Concat(head₀, head₁, ..., headₕ) Wᴼ

其中 headᵢ = Attention(QWqᵢ, KWkᵢ, VWvᵢ)
```

### PyTorch实现

```python
# code/model/attention.py
"""
Multi-Head Attention 实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 可学习的投影矩阵
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """将embedding分割成多个头"""
        # x: (batch, seq_len, d_model)
        # -> (batch, seq_len, num_heads, d_k)
        # -> (batch, num_heads, seq_len, d_k)
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1. 线性投影
        Q = self.W_q(query)  # (batch, seq_len, d_model)
        K = self.W_k(key)
        V = self.W_v(value)

        # 2. 分割成多个头
        Q = self.split_heads(Q, batch_size)  # (batch, num_heads, seq_len_q, d_k)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # 3. 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)

        # 4. 加权求和
        context = torch.matmul(attention_weights, V)
        # context: (batch, num_heads, seq_len, d_k)

        # 5. 合并多头
        context = context.transpose(1, 2).contiguous()
        # -> (batch, seq_len, num_heads, d_k)
        context = context.view(batch_size, -1, self.d_model)
        # -> (batch, seq_len, d_model)

        # 6. 最终投影
        output = self.W_o(context)

        return output, attention_weights


# 示例
d_model = 512
num_heads = 8
batch_size = 2
seq_len = 10

attention = MultiHeadAttention(d_model, num_heads)

query = torch.randn(batch_size, seq_len, d_model)
key = torch.randn(batch_size, seq_len, d_model)
value = torch.randn(batch_size, seq_len, d_model)

output, attn_weights = attention(query, key, value)

print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attn_weights.shape}")
```

## 3.4 注意力模式可视化

```
句子: "The cat sat on the chair because it was tired"

问题: "What is 'it' referring to?"

注意力权重热力图:
          The   cat   sat   on   the  chair becau it  was tired
The       0.1   0.1   0.1   0.1   0.1   0.1   0.1  0.1   0.1  0.1
cat        0.9   0.5   0.3   0.1   0.2   0.3   0.2  0.8   0.3  0.4
sat        0.2   0.4   0.6   0.3   0.1   0.5   0.3  0.2   0.2  0.2
...
it         0.1   0.7   0.2   0.1   0.1   0.2   0.1  0.1   0.1  0.6
                                                ↑
                                          "it"关注"cat"（tired说明是猫累了）

预测: "it" refers to "cat" ✓
```

## 3.5 Transformer的整体架构

```
                    Encoder                         Decoder
                    ┌─────┐                       ┌─────────────┐
                    │Input│                       │   Output    │
                    └──┬──┘                       └──────┬──────┘
                       ↓                              ↑
              ┌─────────────────────┐      ┌────────────────────────┐
              │  Token Embedding    │      │  Token Embedding +     │
              │  + Position Encoding │      │  Position Encoding     │
              └──────────┬──────────┘      └──────────┬─────────────┘
                         ↓                           ↓
              ┌──────────┴───────────────────────────┴──────────┐
              │                    N× layers                      │
              ├────────────────────────────────────────────────────┤
              │                                                    │
              │  ┌──────────┐    ┌──────────┐    ┌──────────┐      │
              │  │  Multi  │ →  │  Feed    │ →  │  Add &   │      │
              │  │  Head   │    │  Forward │    │  Norm    │      │
              │  │Attention│    │ (MLP)    │    │          │      │
              │  └────┬────┘    └──────────┘    └──────┬───┘      │
              │       ↓                              │            │
              │  ┌────┴────┐                  ┌──────┴──────┐    │
              │  │  Add &  │                  │  Multi Head │    │
              │  │  Norm   │                  │  + Encoder  │    │
              │  └────┬────┘                  │  Decoders   │    │
              │       ↓                       └──────┬──────┘    │
              │       └──────────────────────────┬───┘            │
              │                                  ↓                │
              │                            ┌─────┴─────┐          │
              │                            │  Feed     │          │
              │                            │  Forward  │          │
              │                            └─────┬─────┘          │
              │                                  ↓                │
              │                            ┌─────┴─────┐          │
              │                            │  Add &    │          │
              │                            │  Norm     │          │
              │                            └─────┬─────┘          │
              │                                  ↓                │
              └─────────────────────────────────────┘            │
                         ↓                              ↑
              ┌──────────┴──────────┐      ┌──────────┴──────────┐
              │   Linear + Softmax  │      │   Linear + Softmax  │
              └──────────┬──────────┘      └──────────┬──────────┘
                         ↓                              ↓
                    ┌─────┴─────┐                  ┌─────┴─────┐
                    │  Output  │                  │  Output   │
                    │ Sequence │                  │  Tokens  │
                    └──────────┘                  └──────────┘
```

## 3.6 自注意力的计算复杂度

```
O(n² × d)

n: 序列长度
d: embedding维度

相比RNN的 O(n × d²)：
- RNN需要通过隐藏状态传递信息，无法并行
- Self-Attention 可以完全并行，但复杂度是序列长度的平方
```

| 序列长度 | 复杂度 (d=512) |
|----------|----------------|
| 128 | 8M operations |
| 512 | 134M operations |
| 2048 | 2.1B operations |

这就是为什么 **Longformer**、**FlashAttention** 等优化变得重要。

## 3.7 本章小结

1. **RNN的瓶颈**：无法并行，长距离依赖问题
2. **Attention的核心**：Query-Key-Value三角关系
3. **缩放因子 √d_k**：防止点积过大导致梯度消失
4. **多头注意力**：让模型关注多个子空间
5. **复杂度 O(n²)**：这是后续优化的重点

### 思考题

1. 如果把Attention中的缩放因子 √d_k 去掉，会发生什么？
2. 为什么Self-Attention被称为"self"？它和普通的Attention有什么区别？
3. 多头注意力中，如果把所有头合并成一个大矩阵（单头），会有什么区别？

### 延伸阅读

- [Attention is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [FlashAttention: Fast and Memory-Efficient Attention with IO-Awareness (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)
- [ Illustrated: Attention (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)