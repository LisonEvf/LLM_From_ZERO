# 第四章：GPT的诞生 — 语言模型的Scaling Law

## 故事开场：从小到大的飞跃

Bob终于理解了Attention。他兴奋地跟Alice分享："Attention解决了RNN的长距离依赖问题！"

Alice点点头："是的，但还有一个关键问题没有解决——**如何让模型真正理解语言**？"

"RNN也能处理语言啊？"Bob有些困惑。

"试试这个任务："Alice说，"给定一个句子的前半部分，预测下一个词。"

```
输入: "The cat sat on the"
目标: "mat"

RNN的困境:
- 预测"mat"需要理解"cat"和"sat on the"的关系
- 但RNN只能看到之前的信息，无法"展望"未来
- 而且参数少的时候，泛化能力差
```

OpenAI的GPT（Generative Pre-Training）给出了答案：**用大规模无监督预训练学习语言表示，然后用少量标注数据微调**。

## 4.1 GPT的核心思想

GPT的创新可以归结为两点：

### 1. 预训练：学习语言模型

在大规模无标注文本上训练一个**语言模型**：

```
输入: "The cat sat on the mat"
目标: 预测下一个词

Position:   0     1    2    3     4      5
Token:     The  cat  sat  on   the   [MASK]
              ↓    ↓    ↓    ↓     ↓
预测:                               mat  ✓

Loss: -log P("mat" | "The cat sat on the")
```

### 2. 微调：用少量标注数据适配任务

预训练学到了"语言规律"，但每个任务有不同目标。用少量标注数据微调：

```
Task: 情感分类 (正面/负面)
Input: "DeepSeek is great!"
Pre-trained GPT: 学习语言规律
Fine-tuned: 在预训练基础上学习情感分类
Output: Positive ✓
```

## 4.2 GPT-1: 开创性的范式 (2018)

GPT-1的关键数据：

| 项目 | 数值 |
|------|------|
| 参数量 | 1.17亿 |
| 预训练数据量 | 约5GB (BooksCorpus) |
| 层数 | 12 |
| 注意力头数 | 12 |
| 上下文长度 | 512 |

架构：仅用**解码器**（Decoder-only）Transformer

```
              Input: "The cat sat on"
                 ↓
         Token Embedding + Position
                 ↓
         ┌───────────────────────┐
         │   12 Transformer Blocks │
         │   (Decoder-only)       │
         └───────────────────────┘
                 ↓
              LayerNorm
                 ↓
           Linear → Softmax
                 ↓
        输出: 下一个词的概率分布
```

**为什么用Decoder-only？**

因为语言模型的任务是**预测下一个词**，只需要看到左侧上下文（单向），不需要右侧信息。

## 4.3 GPT-2: Zero-Shot的突破 (2019)

GPT-2的核心发现：**语言模型足够大时，可以在没有任何微调的情况下泛化到新任务**。

这叫 **Zero-Shot** 学习：

```
Prompt: "Translate to French: English sentence"
Input:  "The cat sat on the mat"
Model:  "Le chat s'est assis sur le tapis"  ← 模型自己完成翻译！

不需要任何翻译样本的微调
```

GPT-2的关键改进：

| 方面 | GPT-1 | GPT-2 |
|------|-------|-------|
| 参数量 | 117M | 1.5B (15亿) |
| 词表大小 | 40k | 50k (BPE) |
| 上下文 | 512 | 1024 |
| 预训练数据 | BooksCorpus | WebText (800万网页) |

GPT-2的训练目标不是"做任务"，而是"预测下一个词"。但在足够大的规模下，模型学会了做各种任务。

## 4.4 GPT-3: Few-Shot的震惊 (2020)

GPT-3证明了 **Scaling Law**：模型越大，能力越强。

| 指标 | GPT-3 |
|------|-------|
| 参数量 | 1750亿 |
| 上下文长度 | 2048 |
| 训练数据 | 45TB (Common Crawl) |
| 参数存储 | FP16需要350GB |

### Few-Shot Learning

```
Prompt (包含几个例子):
"translate English to French:

English: The weather is nice
French: Le temps est agréable

English: I love reading
French:"

Model output: "J'adore lire"  ← 零样本无法完成，但Few-Shot可以
```

Few-Shot提供几个示例（k-shot），模型从中学习模式。

### GPT-3的能力

在57个基准测试上，GPT-3在Many-shot设置下超越了当时最好的微调模型：

```
- 自然语言推理: 89% (超越人类基线)
- 问答: 80%+
- 数学: 30% (仍然有限)
- 代码: ...
```

## 4.5 Scaling Law

OpenAI的论文 "Scaling Laws for Neural Language Models" (2020) 揭示了重要规律：

```
Loss ∝ N^(-0.076)

N = 参数数量

模型越大，困惑度（perplexity）越低
```

实验数据：

| 参数规模 | 困惑度 |
|----------|--------|
| 10M | 45.2 |
| 100M | 24.2 |
| 1B | 18.5 |
| 10B | 14.5 |
| 100B | 12.5 |
| 175B | 11.5 |

**关键洞察**：双倍参数 ≈ 类似双倍数据

DeepMind的Chinchilla (2022) 提出了不同的观点：

> GPT-3用1.7T token训练了175B参数，但理论上相同的计算预算可以用更多token训练更小的模型。

Chinchilla建议：**模型大小 × Token数量 ≈ 常数**

GPT-3: 175B × 300B tokens → 52.5T
Chinchilla: 70B × 1.4T tokens → 98T (更优)

## 4.6 GPT的架构细节

### Transformer Decoder Block

```python
# code/model/gpt_block.py
"""
GPT的Transformer Block (Decoder-only)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GPTBlock(nn.Module):
    """单个GPT Transformer Block"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # GPT-2使用GELU
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Self-Attention with Pre-Norm
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)

        # FFN with Pre-Norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class GPTModel(nn.Module):
    """完整的GPT模型"""
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, dropout=0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        # Token嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # 位置嵌入 (可学习)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # 输出头
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 权重绑定 (可选)
        # self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids, position_ids=None, attention_mask=None):
        """
        参数:
            input_ids: (batch, seq_len)
            position_ids: 可选，默认自动创建
        返回:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape

        # Token嵌入
        token_emb = self.token_embedding(input_ids)

        # 位置嵌入
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device)
        position_emb = self.position_embedding(position_ids)

        # 组合
        x = token_emb + position_emb

        # 通过Transformer blocks
        for block in self.blocks:
            x = block(x, mask=attention_mask)

        x = self.norm(x)

        # 输出 logits
        logits = self.lm_head(x)

        return logits

    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        """自回归生成"""
        for _ in range(max_new_tokens):
            # 截断到最大长度
            input_ids_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]

            # 前向传播
            logits = self.forward(input_ids_cond)

            # 取最后一个位置的logits
            logits = logits[:, -1, :] / temperature

            # Top-k 过滤
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')

            # 采样
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 拼接
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
```

## 4.7 GPT-2的配置对比

| 模型 | 层数 | 维度 | 头数 | 参数量 |
|------|------|------|------|--------|
| GPT-2 Small | 12 | 768 | 12 | 117M |
| GPT-2 Medium | 24 | 1024 | 16 | 345M |
| GPT-2 Large | 36 | 1280 | 20 | 774M |
| GPT-2 XL | 48 | 1600 | 25 | 1.5B |

## 4.8 本章小结

1. **GPT的开创性**：预训练+微调的范式
2. **Decoder-only**：语言模型只需要单向注意力
3. **Zero-Shot/Few-Shot**：大模型可以不微调就泛化
4. **Scaling Law**：模型越大，性能越好
5. **挑战**：推理成本随着模型增大而指数增长

### 思考题

1. 为什么GPT使用Decoder-only，而原始Transformer使用Encoder-Decoder？
2. 如果一个模型无限大，Few-Shot能力会无限提升吗？
3. GPT-3的175B参数用FP16存储需要350GB显存，如何在消费级GPU上运行？

### 延伸阅读

- [GPT-1: Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [GPT-2: Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
- [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)