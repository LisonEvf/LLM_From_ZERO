# 第六章：高效之道 — MoE专家混合

## 故事开场：计算的浪费

Bob在运行GPT-3时发现一个惊人的事实：

```
GPT-3: 175B参数
处理1个token: 需要计算 175B 次浮点运算 (FLOPs)
实际激活: 所有参数都在计算

但：真的需要所有参数吗？
```

**答案是：不需要。**

语言模型中，不是所有参数对每个token都有同样重要的贡献。"Deep"这个词的语义，主要由某些"专家"（experts）处理，而不是全部参数。

这就是 **MoE（Mixture of Experts）** 的核心思想：**让模型学会"分工"**。

## 6.1 MoE的基本原理

### 稠密模型 vs 稀疏模型

```
稠密模型 (Dense):
┌─────────────────────────────────────┐
│  所有参数都参与每个token的计算       │
│                                     │
│  Input → [████████████] → Output    │
│            所有参数都激活            │
└─────────────────────────────────────┘

稀疏模型 (MoE):
┌─────────────────────────────────────┐
│  只有"专家"被激活，其他静默          │
│                                     │
│  Input → 门控 → [专家1]             │
│              ↓                      │
│           [专家3] → Output          │
│              ↓                      │
│           [专家5]                   │
└─────────────────────────────────────┘
```

### Switch Transformer的创新

2021年，Google提出了 **Switch Transformer**，实现了一个关键突破：

> **每次只激活1个专家（top-1 routing）**，而不是之前的top-k多个专家。

```
之前MoE:
门控 → [专家1] [专家2] [专家3] [专家4]  (top-2激活)

Switch Transformer:
门控 → [专家3]  (只有1个被选中！)
```

这大大降低了**通信开销**和**计算量**。

## 6.2 门控机制

门控（Gating）是MoE的核心，决定哪个专家处理当前token。

```python
# code/model/moe.py
"""
MoE门控机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TopKGating(nn.Module):
    """Top-K门控，选择最相关的K个专家"""
    def __init__(self, d_model, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 门控网络
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        """
        参数:
            x: (batch, seq_len, d_model)
        返回:
            dispatch: (batch, seq_len, num_experts) - 软路由权重
            top_k_indices: (batch, seq_len, top_k) - 激活的专家索引
        """
        # 计算每个专家的logits
        logits = self.gate(x)  # (batch, seq_len, num_experts)

        # 获取top-k
        top_logits, top_indices = torch.topk(logits, self.top_k, dim=-1)

        # 计算softmax权重
        top_k_weights = F.softmax(top_logits, dim=-1)

        # 创建稀疏路由矩阵
        dispatch = torch.zeros_like(logits)
        dispatch.scatter_(-1, top_indices, top_k_weights)

        return dispatch, top_indices, top_k_weights


# 示例
num_experts = 8
top_k = 2
d_model = 512

gating = TopKGating(d_model, num_experts, top_k)

x = torch.randn(2, 10, d_model)  # batch=2, seq=10
dispatch, top_indices, weights = gating(x)

print(f"Dispatch shape: {dispatch.shape}")  # (2, 10, 8)
print(f"Top indices: {top_indices[0, 0]}")  # e.g., tensor([3, 5])
print(f"Top weights: {weights[0, 0]}")  # e.g., tensor([0.7, 0.3])
```

## 6.3 专家结构

每个专家本质上是一个**前馈网络**（FFN）：

```
专家网络:
Input → Linear(d_model, d_ff) → GELU → Linear(d_ff, d_model) → Output

等价于标准Transformer中的FFN块
```

## 6.4 DeepSeek-MoE的创新

DeepSeek在此基础上做了进一步优化：

### 1. 细粒度专家分割

```
传统MoE:        4个专家，每个专家 d_ff=2048
                总参数量 = 4 × 2048 = 8192

DeepSeek MoE:    16个专家，每个专家 d_ff=512
                总参数量 = 16 × 512 = 8192 (相同)

但细粒度专家增加了灵活性：
- 更精细的知识分解
- 每个token可以激活更多小专家
```

### 2. 共享专家隔离

```
普通专家: 处理特定领域知识
共享专家: 处理通用知识（如语法、常识）

总专家数 = 共享专家 + 路由专家
```

### 3. Top-K 激活

```python
# DeepSeek-MoE 配置
num_experts = 64
top_k = 6  # 激活6个专家（2个共享 + 4个路由）

# 共享专家始终激活
# 路由专家根据门控选择top-4
```

## 6.5 MoE的计算效率

```
假设:
- 模型总参数量: 1T (1万亿)
- 专家数: 64
- top_k: 4

活跃参数量/Token = (64 - 共享数)×(d_ff/总参数) × top_k

对比:
- 稠密模型: 1T 参数全部激活
- MoE模型: 约 200B 参数激活 (活跃比例 20%)
- 理论加速: 约 5×
```

### 通信开销

分布式MoE的挑战：**不同token激活不同专家，需要通信**。

```
Data Parallel: 每个GPU处理不同的batch，但激活所有专家
Model Parallel: 不同专家在不同GPU，但每个token需要访问多个GPU

Switch Transformer: 增加"容量因子"减少通信
```

## 6.6 从零实现MoE Transformer Block

```python
# code/model/moe_transformer.py
"""
MoE Transformer Block 实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Expert(nn.Module):
    """单个专家网络"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.ffn(x)


class MoELayer(nn.Module):
    """MoE层"""
    def __init__(self, d_model, d_ff, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 专家列表
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff) for _ in range(num_experts)
        ])

        # 门控
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        # 门控
        logits = self.gate(x)  # (batch, seq_len, num_experts)
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)

        # softmax权重
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        # 初始化输出
        output = torch.zeros_like(x)

        # 对每个token，处理其激活的专家
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, :, i]  # (batch, seq_len)
            weight = top_k_weights[:, :, i].unsqueeze(-1)  # (batch, seq_len, 1)

            # 收集该专家的输出
            for expert_id in range(self.num_experts):
                mask = (expert_idx == expert_id)  # (batch, seq_len)
                if mask.any():
                    expert_input = x[mask]  # (active_tokens, d_model)
                    expert_output = self.experts[expert_id](expert_input)
                    output[mask] += expert_output * weight[mask]

        return output


class MoETransformerBlock(nn.Module):
    """包含MoE的Transformer Block"""
    def __init__(self, d_model, num_heads, d_ff, num_experts, top_k, dropout=0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.moe = MoELayer(d_model, d_ff, num_experts, top_k)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Self-Attention
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)

        # MoE FFN
        moe_out = self.moe(x)
        x = self.norm2(x + moe_out)

        return x


# 示例
d_model = 512
num_heads = 8
d_ff = 2048
num_experts = 8
top_k = 2

block = MoETransformerBlock(d_model, num_heads, d_ff, num_experts, top_k)

x = torch.randn(2, 10, d_model)
output = block(x)

print(f"Output shape: {output.shape}")
```

## 6.7 MoE的负载均衡

MoE的一个关键挑战：**如何避免某些专家被过度使用**？

```
问题: 门控可能总是选择最强的专家，导致其他专家"失业"
```

解决方案：**辅助损失（Auxiliary Loss）**

```python
def load_balancing_loss(gate_logits, top_k_indices, num_experts):
    """
    负载均衡损失
    鼓励均匀分配token到各个专家
    """
    # 每个专家被选中的频率
    expert_counts = torch.zeros(num_experts)
    for expert_id in range(num_experts):
        expert_counts[expert_id] = (top_k_indices == expert_id).sum()

    # 频率
    expert_frequency = expert_counts / top_k_indices.numel()

    # 门控概率
    gate_probs = F.softmax(gate_logits, dim=-1).mean(dim=[0, 1])

    # 损失 = sum(frequency * gate_prob)
    # 最小化这个损失 → 均匀分配
    loss = num_experts * torch.sum(gate_probs * expert_frequency)

    return loss
```

## 6.8 MoE vs 稠密模型

| 方面 | 稠密模型 | MoE |
|------|----------|-----|
| 参数总量 | N | N (可能更大) |
| 活跃参数/Token | N | N × (top_k / num_experts) |
| 显存占用 | 高 | 中（参数量大，但激活少） |
| 计算量 | O(N) | O(N × top_k / num_experts) |
| 通信量 | 低 | 高（分布式） |
| 效果 | 基准 | 相当或更好 |

## 6.9 本章小结

1. **MoE思想**：稀疏激活，只用部分参数处理每个token
2. **门控机制**：选择top-k专家处理输入
3. **Switch Transformer**：top-1 routing降低通信开销
4. **DeepSeek-MoE**：细粒度专家分割 + 共享专家
5. **负载均衡**：辅助损失避免专家失效

### 思考题

1. MoE的门控网络是如何学习的？梯度如何回传？
2. 为什么说MoE的通信开销是分布式训练的主要瓶颈？
3. 如果一个专家被选中的概率远高于其他专家，会发生什么？

### 延伸阅读

- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)
- [ST-MoE: Stable and Transferable Mixture-of-Experts](https://arxiv.org/abs/2202.08906)
- [DeepSeek-MoE: Towards Ultimate Specialization in Mixture-of-Experts](https://arxiv.org/abs/2401.14166)