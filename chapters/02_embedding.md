# 第二章：意义的表示 — 从词嵌入到上下文表示

## 故事开场：Bob的词向量困惑

Bob学会了Tokenizer，他把"DeepSeek"转换成了 `[68, 256, 259]`。但他突然意识到一个问题：

**这些数字本身有什么含义吗？**

`68` 代表字母 "D"，这只是一种编码约定。模型要真正理解"DeepSeek"这个词，需要一种**有意义**的表示方式。

"king - man + woman ≈ queen" —— 这个著名的词向量等式让Bob着迷。如果词可以用数字向量表示，而且这些向量之间还有语义关系，那该多好啊！

但问题是：**一个词在不同的上下文里，意思可能完全不同。**

"bank" 可以是银行，也可以是河岸。"苹果"可以是水果，也可以是公司。

如何让词的表示包含上下文？让我们跟随Bob的探索之旅。

## 2.1 早期的词向量：Word2Vec

2013年，Tomas Mikolov 发表了 Word2Vec，提出了**分布式假设**：

> "You shall know a word by the company it keeps" (Firth, 1957)

简单说就是：**出现在相似上下文中的词，语义也相似。**

### Skip-gram 模型

Word2Vec 的 Skip-gram 模型通过预测上下文来学习词向量：

```
给定中心词 "deep"，预测周围词 ["is", "great", "and"]
      center        context
        ↓              ↑
       [deep]  →  [is] [great] [and]
         ↓
      Embedding
         ↓
    词向量: [0.2, -0.5, 0.8, ...] (300维)
```

训练目标：最大化条件概率 $P(context|word)$

### 经典例子：king - man + woman ≈ queen

这个著名的等式之所以成立，是因为：

```
vec("king") - vec("man") + vec("woman") ≈ vec("queen")

king                          queen
  ↓                            ↑
man → woman (类比关系)  king - man + woman
```

在向量空间中，"性别"是一个维度。"king" 和 "queen" 在这个维度上相近，而 "man" 和 "woman" 是对应的男性/女性版本。

## 2.2 静态嵌入的问题：一词多义

但 Word2Vec 有一个致命问题：**每个词只有一个向量**。

让我们看一个例子：

```
句子1: "The bank is closed" (银行关门了)
句子2: "The river bank is beautiful" (河岸很美)

银行         河岸
  ↓           ↓
[0.2, ...]  [0.7, ...]
  ↓           ↓
 "bank"共享同一个词向量 [0.5, ...]
```

无论是"银行"还是"河岸"，都用同一个向量表示。这导致：
- 句子1的"bank"语义被稀释
- 句子2的"bank"语义被污染

## 2.3 上下文嵌入：Transformer的突破

2017年，Transformer 问世。它用**自注意力机制**让每个词的表示**取决于上下文**。

```
输入: "The bank is closed"
         ↓
      Tokenize
         ↓
      [The] [bank] [is] [closed]
         ↓
      通过自注意力，每个词"看到"其他词
         ↓
      bank 的表示 = f(bank, The, is, closed)
                  考虑了上下文！
```

这样，"银行"和"河岸"的"bank"就有了不同的向量表示。

## 2.4 位置编码：让序列有序

Transformer 的自注意力机制是**位置无关**的——它不知道"bank"在句子的哪个位置。

为了让模型知道位置信息，需要**显式注入位置编码**：

```
原始输入: [The] [bank] [is] [closed]
           ↓        ↓      ↓      ↓
位置编码:   0        1      2      3
           ↓        ↓      ↓      ↓
最终输入: [The+P0] [bank+P1] [is+P2] [closed+P3]
           ↓        ↓      ↓      ↓
        词向量 + 位置向量 = 输入表示
```

### 原始Transformer：Sinusoidal位置编码

原始论文使用了正弦/余弦函数：

```python
import numpy as np

def get_sinusoidal_position_encoding(seq_len, d_model):
    """生成sinusoidal位置编码"""
    PE = np.zeros((seq_len, d_model))

    # pos: 位置 (0, 1, 2, ...)
    # i: 维度索引 (0, 1, 2, ..., d_model/2)
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            # 偶数维度用sin
            PE[pos, i] = np.sin(pos / 10000 ** (i / d_model))
            # 奇数维度用cos
            PE[pos, i + 1] = np.cos(pos / 10000 ** (i / d_model))

    return PE

# 示例：位置0和位置1的编码
pe = get_sinusoidal_position_encoding(10, 8)
print(f"位置0: {pe[0]}")
print(f"位置1: {pe[1]}")
```

### 特点：
- **可外推**：理论上可以处理任意长度的序列
- **相对位置友好**：sin(pos/k) 和 sin((pos+Δ)/k) 包含相对位置 Δ 的信息

## 2.5 从零实现词嵌入

下面是一个简化的词嵌入实现：

```python
# code/model/embedding.py
"""
词嵌入与位置编码的简化实现
用于教学目的
"""

import numpy as np


class TokenEmbedding:
    """可学习的Token嵌入"""
    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        # 随机初始化
        self.W = np.random.randn(vocab_size, d_model) * 0.02

    def __call__(self, token_ids):
        """根据token ids获取嵌入向量"""
        return self.W[token_ids]

    def forward(self, token_ids):
        return self(token_ids)


class SinusoidalPositionEncoding:
    """Sinusoidal位置编码（非可学习）"""
    def __init__(self, d_model, max_len=5000):
        self.d_model = d_model
        self.PE = self._generate_encoding(max_len)

    def _generate_encoding(self, max_len):
        PE = np.zeros((max_len, self.d_model))
        for pos in range(max_len):
            for i in range(0, self.d_model, 2):
                # 频率从 1/10000^0 到 1/10000^(d_model/2)
                denominator = 10000 ** (i / self.d_model)
                PE[pos, i] = np.sin(pos / denominator)
                if i + 1 < self.d_model:
                    PE[pos, i + 1] = np.cos(pos / denominator)
        return PE

    def __call__(self, seq_len):
        return self.PE[:seq_len]


# 示例：组合使用
vocab_size = 50000
d_model = 512
seq_len = 10

token_emb = TokenEmbedding(vocab_size, d_model)
pos_enc = SinusoidalPositionEncoding(d_model)

# 输入token ids
token_ids = [2045, 3086, 1996, 2003]  # "The bank is"

# 获取嵌入
token_embeddings = token_emb(token_ids)  # Shape: (4, 512)
position_encoding = pos_enc(len(token_ids))  # Shape: (4, 512)

# 组合：嵌入 + 位置编码
input_representations = token_embeddings + position_encoding

print(f"Token嵌入: {token_embeddings.shape}")
print(f"位置编码: {position_encoding.shape}")
print(f"最终输入: {input_representations.shape}")
```

## 2.6 可视化：词嵌入的空间

```
                    语义空间 (2D投影)

           queen
             ↑
            /
    man ----•---- woman
           /
    king
             |
             |
    ┌─────────────────────────────────────┐
    │          feminine (+1)              │
    └─────────────────────────────────────┘
             |
             |
    ┌─────────────────────────────────────┐
    │          masculine (-1)             │
    └─────────────────────────────────────┘
             |
             |
           doctor
             ↑
            /
    man ----•---- woman
           /
    nurse
```

**观察**：
1. `king` 和 `queen` 在"性别"维度上形成对比
2. `man` 和 `woman` 形成另一组对比
3. 类比关系：`king:man = queen:woman`

## 2.7 上下文长度与嵌入质量

嵌入的质量与上下文长度密切相关：

| 方法 | 上下文范围 | 表示方式 | 典型模型 |
|------|------------|----------|----------|
| Word2Vec | 无（静态） | 单一向量 | - |
| ELMo | 双向LSTM | 两层BiLSTM | ELMo |
| BERT | 完整上下文 | Transformer | BERT, RoBERTa |
| GPT | 单向（左侧） | Transformer | GPT-2, GPT-3 |

### GPT的单向性

GPT 使用**单向**（从左到右）注意力：

```
输入: "I love DeepSeek"
         ↓
预测下一个词时，只能看到左边的词
"I" → 看 "I" 的嵌入
"love" → 看 "I love" 的嵌入
"DeepSeek" → 看 "I love DeepSeek" 的嵌入
```

这导致 GPT 的嵌入是**条件概率**的，而 BERT 的嵌入是**联合概率**的。

## 2.8 嵌入的实践应用

### 词相似度

```python
import numpy as np
from numpy.linalg import norm

def cosine_similarity(a, b):
    """计算两个向量的余弦相似度"""
    return np.dot(a, b) / (norm(a) * norm(b))

# 假设这些是从训练好的模型中得到的词向量
vec_king = np.array([0.5, 0.3, 0.8, 0.1])
vec_queen = np.array([0.6, 0.2, 0.9, 0.15])
vec_apple = np.array([0.1, -0.5, 0.3, 0.9])

print(f"king vs queen: {cosine_similarity(vec_king, vec_queen):.4f}")
# 输出: ~0.98 (高相似度)

print(f"king vs apple: {cosine_similarity(vec_king, vec_apple):.4f}")
# 输出: ~0.30 (低相似度)
```

### 词类比

```python
def word_analogy(word_a, word_b, word_c, embeddings):
    """
    类比: a is to b as c is to ?

    king - man + woman ≈ queen
    实际上: king + (woman - man) ≈ queen
    """
    vec_a = embeddings[word_a]
    vec_b = embeddings[word_b]
    vec_c = embeddings[word_c]

    # 计算目标向量
    vec_target = vec_a - vec_b + vec_c

    # 找最接近的词（简化实现）
    best_match = None
    best_sim = -1
    for word, vec in embeddings.items():
        if word in [word_a, word_b, word_c]:
            continue
        sim = cosine_similarity(vec_target, vec)
        if sim > best_sim:
            best_sim = sim
            best_match = word

    return best_match, best_sim
```

## 2.9 本章小结

1. **词嵌入将词转换为有语义含义的向量**
2. **Word2Vec基于分布式假设：相似上下文的词语义相似**
3. **静态嵌入无法处理一词多义**
4. **Transformer通过自注意力实现上下文相关的嵌入**
5. **位置编码注入序列位置信息**

### 思考题

1. 为什么Word2Vec的"king - man + woman ≈ queen"能在向量空间成立？这体现了什么语言学假设？
2. 如果把位置编码改为**可学习的**，与Sinusoidal编码相比，有什么优缺点？
3. GPT使用单向注意力，而BERT使用双向。这对它们的嵌入表示有什么影响？

### 延伸阅读

- [Efficient Estimation of Word Representations in Vector Space (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781)
- [Attention is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Contextual Word Representations: A Brief History (Salander, 2023)](https://arxiv.org/abs/2304.00675)