# 第十章：汇聚一切 — DeepSeek风格完整模型

## 故事开场：集大成者

Alice回顾了她学到的所有技术：

```
第一章: Tokenizer (BPE) → 如何把文字变成数字
第二章: Embedding → 如何表示词的语义
第三章: Attention → 如何让词相互"看到"对方
第四章: GPT → 如何从语言模型到生成模型
第五章: RoPE → 如何处理超长上下文
第六章: MoE → 如何高效计算
第七章: 量化 → 如何压缩模型
第八章: RAG → 如何访问外部知识
第九章: Function Calling → 如何执行动作
```

"现在，"Bob说，"让我们看看这些技术如何组合成现代LLM。DeepSeek-V4就是这种集大成的代表。"

## 10.1 DeepSeek技术栈总览

```
┌─────────────────────────────────────────────────────────────────┐
│                    DeepSeek-V4 架构                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  输入: "请用Python写一个快速排序"                                │
│         ↓                                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Tokenizer (BPE)                       │   │
│  │              SentencePiece, 32K 词表                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│         ↓                                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 Input Embedding                          │   │
│  │  Token Embedding + RoPE Position Encoding                │   │
│  └─────────────────────────────────────────────────────────┘   │
│         ↓                                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Transformer Layers (32层)                   │   │
│  │                                                          │   │
│  │   ┌─────────────────────────────────────────────────┐  │   │
│  │   │              MoE (稀疏专家)                       │  │   │
│  │   │  - 共享专家 (始终激活)                             │  │   │
│  │   │  - 路由专家 (top-k选择)                           │  │   │
│  │   │  - 每个Token激活部分参数                          │  │   │
│  │   └─────────────────────────────────────────────────┘  │   │
│  │                                                          │   │
│  │   ┌─────────────────────────────────────────────────┐  │   │
│  │   │              Self-Attention (RoPE)              │  │   │
│  │   │  - 旋转位置编码，支持长上下文                     │  │   │
│  │   │  - 稀疏注意力（可选）                            │  │   │
│  │   └─────────────────────────────────────────────────┘  │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│         ↓                                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 Output Projection                        │   │
│  │              Logits → 下一个Token概率                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│         ↓                                                        │
│  输出: "def quick_sort(arr): ..." (生成的代码)                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 10.2 各组件配合示例

### 编码阶段

```python
# 用户输入
text = "请用Python写一个快速排序"

# 1. Tokenizer (BPE)
tokens = tokenizer.encode(text)
# tokens: [2045, 3086, 1996, 2003, 2613, 1234, ...]
print(f"Token数量: {len(tokens)}")  # 远少于字符数

# 2. Embedding + RoPE
token_embeddings = token_embedding(tokens)
position_ids = torch.arange(len(tokens))
position_embeddings = rope(position_ids)

# 组合
input_representations = token_embeddings + position_embeddings
```

### 计算阶段

```python
# 3. 通过Transformer Layers
for layer in transformer_layers:
    # Self-Attention with RoPE
    attn_output = layer.attention(input_representations, rope_cos_sin)

    # MoE FFN
    moe_output = layer.moe(attn_output)

    # 残差连接和LayerNorm
    input_representations = layer.norm(attn_output + moe_output)
```

### 解码阶段

```python
# 4. 输出
logits = output_projection(input_representations[:, -1, :])
next_token_probs = F.softmax(logits, dim=-1)

# 采样（或贪婪/Beam Search）
next_token = torch.multinomial(next_token_probs, num_samples=1)
generated_tokens.append(next_token.item())

# 重复直到生成结束
# ...
```

## 10.3 DeepSeek-MoE的关键设计

### 专家配置

```python
# DeepSeek-V3 MoE 配置示例
config = {
    "num_experts": 64,           # 总专家数
    "num_active_experts": 6,      # 每个Token激活6个
    "shared_experts": 2,          # 2个共享专家始终激活
    "routed_experts": 4,          # 从60个路由专家中选4个
}

# 路由专家选择
def moe_routing(token_emb, gate):
    # 计算门控logits
    gate_logits = gate(token_emb)  # (batch, seq, num_experts)

    # Top-K选择
    top_k_logits, top_k_indices = torch.topk(gate_logits, k=4, dim=-1)

    # Softmax权重
    weights = F.softmax(top_k_logits, dim=-1)

    # 激活的专家
    active_experts = [experts[i] for i in top_k_indices[0]]

    return active_experts, weights
```

### 计算效率分析

```
假设: 总参数量 = 1T (1万亿)
      专家数 = 64
      top_k = 6

每个Token活跃参数 = (64 * d_ff) * (6/64) = 6 * d_ff

对比稠密模型:
- 稠密: 所有参数都计算
- MoE: 约 6/64 = 9.4% 的FFN参数活跃

稀疏度: 90.6%
计算量节省: ~10×
```

## 10.4 长上下文处理

DeepSeek使用RoPE + 一些优化来处理长上下文：

```python
def long_context_attention(query, key, value, position_ids):
    """
    长上下文注意力

    1. RoPE编码位置
    2. 分块处理避免O(n²)显存
    3. FlashAttention加速
    """
    # 应用RoPE
    query = apply_rope(query, position_ids)
    key = apply_rope(key, position_ids)

    # 分块计算（FlashAttention自动做）
    # 但如果上下文特别长，可以使用稀疏注意力模式

    return flash_attention(query, key, value)
```

### 上下文长度扩展

```
DeepSeek-V2: 上下文 128K tokens
DeepSeek-V3: 上下文 256K+ tokens

扩展方法:
1. YaRN: 对RoPE频率进行缩放
2. NTK-aware: 根据位置动态调整缩放
3. LongBench: 测试长上下文的标准
```

## 10.5 量化与推理优化

DeepSeek通常使用INT4/INT8量化来部署：

```python
# 量化配置
quant_config = {
    "weight_bits": 4,           # 权重4位量化
    "activation_bits": 8,        # 激活8位量化
    "group_size": 128,           # 每组128个参数共享scale
    "method": "GPTQ",            # 或 AWQ
}

# 量化后的模型
quantized_model = load_quantized_model("deepseek-v3-4bit")

# 推理时使用INT运算加速
# 核心计算使用INT4/INT8，而非FP16
```

### 推理优化技术

| 技术 | 作用 | 效果 |
|------|------|------|
| FlashAttention | 减少HBM读写 | 2-4×加速 |
| PagedAttention | 优化KV cache | 内存利用率↑ |
| 量化 | 降低计算精度 | 显存↓，速度↑ |
| 连续批处理 | 动态batch | 吞吐量↑ |

## 10.6 完整代码示例

```python
# code/model/deepseek.py
"""
DeepSeek风格模型（简化版）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DeepSeekMoE(nn.Module):
    """DeepSeek MoE层"""
    def __init__(self, d_model, d_ff, num_experts, top_k, num_shared=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_shared = num_shared

        # 共享专家（始终激活）
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_shared)
        ])

        # 路由专家
        self.routed_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])

        # 门控
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        # 1. 门控计算
        gate_logits = self.gate(x)  # (batch, seq_len, num_experts)
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        # 2. 激活的路由专家
        output = torch.zeros_like(x)

        # 处理每个token
        for i in range(batch):
            for j in range(seq_len):
                token_x = x[i, j:j+1]  # (1, d_model)

                # 收集激活的专家输出
                expert_outputs = []
                for k in range(self.top_k):
                    expert_id = top_k_indices[i, j, k].item()
                    weight = top_k_weights[i, j, k]

                    expert_out = self.routed_experts[expert_id](token_x)
                    expert_outputs.append(expert_out * weight)

                # 加上共享专家的输出
                for shared_exp in self.shared_experts:
                    output[i, j] += shared_exp(token_x).squeeze(0)

                # 路由专家加权求和
                output[i, j] += sum(expert_outputs).squeeze(0)

        return output


class DeepSeekBlock(nn.Module):
    """DeepSeek Transformer Block"""
    def __init__(self, d_model, num_heads, d_ff, num_experts, top_k):
        super().__init__()

        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        self.moe = DeepSeekMoE(d_model, d_ff, num_experts, top_k)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Self-Attention
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)

        # MoE FFN
        moe_out = self.moe(x)
        x = self.norm2(x + moe_out)

        return x


class DeepSeekModel(nn.Module):
    """完整的DeepSeek模型"""
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff,
                 num_experts, top_k, max_seq_len):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            DeepSeekBlock(d_model, num_heads, d_ff, num_experts, top_k)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        # Embedding
        token_emb = self.token_embedding(input_ids)
        position_emb = self.position_embedding(
            torch.arange(input_ids.size(1), device=input_ids.device)
        )
        x = token_emb + position_emb

        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask=attention_mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits


# 示例配置
config = {
    "vocab_size": 32000,
    "d_model": 2048,
    "num_heads": 16,
    "num_layers": 32,
    "d_ff": 8192,
    "num_experts": 64,
    "top_k": 6,
    "max_seq_len": 4096,
}

model = DeepSeekModel(**config)
print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
```

## 10.7 本章小结

1. **集大成设计**：DeepSeek整合了BPE、RoPE、MoE、Attention等所有技术
2. **MoE稀疏激活**：只激活部分专家，大幅降低计算量
3. **RoPE长上下文**：支持128K+ tokens的上下文
4. **量化压缩**：INT4/INT8量化降低部署成本
5. **系统协同**：每个组件都经过优化，相互配合

### 思考题

1. DeepSeek的MoE和标准Transformer的FFN有什么本质区别？各自的优势是什么？
2. 如果要在消费级GPU上运行DeepSeek模型，需要哪些优化？
3. 为什么DeepSeek选择共享专家+路由专家的设计？这与纯路由MoE相比有什么优势？

### 延伸阅读

- [DeepSeek-V2: A Strong Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [The MoE Architecture: From Principles to Implementation (Blog)](https://arxiv.org/abs/2405.04434)