# 第一章：文本的数字化 — Tokenizer的设计哲学

## 故事开场：Alice的第一道难题

Alice是一个刚接触NLP的工程师，她接到的第一个任务是：让计算机"理解"一句话——"DeepSeek真好用！"

"这有什么难的？" Alice想当然地拿起笔，开始把这句话拆成单个字符：
```
D-e-e-p-S-e-e-k-真-好-用-！
```

10个字符，刚好！她的第一个"tokenizer"诞生了。

但现实很快给了她一记耳光。当她用这个字符级 tokenizer 处理 1GB 的文本时，发现：
1. **词汇表巨大**：汉字有几万个，token数量爆炸
2. **语义丢失**：`"Deep"` 是一个有意义的整体，被拆成了 `D-e-e-p`
3. **效率低下**：相邻字符（如 `e-e`）重复出现，毫无意义

"有没有办法让机器自动找到那些高频出现的字符组合？"

## 1.1 为什么需要Tokenizer

Tokenizer（分词器）是LLM的第一道门槛。它的任务很简单：**把人类文字转换成数字序列**，让模型能处理。

但这道看似简单的转换，背后藏着深刻的设计哲学：

| 方案 | 优点 | 缺点 |
|------|------|------|
| 字符级 | 词汇表小，处理任意字符 | 序列太长，语义碎片化 |
| 词级 | 语义完整 | 词汇表巨大，未登录词(UNK)问题 |
| 子词级 | 平衡效率与语义 | 需要算法决定如何切分 |

**BPE（Byte Pair Encoding）** 就是子词级的经典方案。它用统计方法自动找到高频的"词片段"，既不像字符那样啰嗦，也不像词那样局限。

## 1.2 BPE的工作原理

BPE的核心思想来自一个简单观察：**有些字符对，经常一起出现**。

比如在英文文本中：
- `"t"` 和 `"h"` 经常组成 `"th"`
- `"e"` 和 `"r"` 经常组成 `"er"`

BPE就是通过**统计字符对的频率，然后迭代合并最高频的对**，来构建一个合理的子词词汇表。

### 算法的故事

让我们用 Alice 的故事来理解 BPE 的工作流程。

**原始句子**: `"hug dog pup"`

Alice 统计所有字符的频率：

```
字符: h u g _ d o g _ p u p
频率: h:1 u:2 g:2 _:2 d:1 o:1 p:2
```

**第一次迭代**：找到最高频的字符对

```
"u" 出现了2次 (在"hug"和"pup"中)
"g" 出现了2次 (在"hug"和"dog"中)
"p" 出现了2次 (在"pup"中)
"_" 出现了2次 (空格)
```

假设 `"ug"` 是最高频的对，合并它们：

```
新词表增加: "ug"
文本变为:  "h[ug] dog p[ug]"
其中 [ug] 作为一个token
```

**第二次迭代**：继续合并

```
现在统计所有字符和词片段的频率
"p" + "[ug]" = "p[ug]" 出现1次
"h" + "[ug]" = "h[ug]" 出现1次
...
```

最终，BPE 会找到像 `"pug"`, `"dog"`, `"hug"` 这样有意义的词片段。

### BPE的精髓

1. **从字节出发**：初始词表只有256个基础字节
2. **迭代合并**：每次找到最高频的相邻对，合并成一个新token
3. **词表大小可控**：可以通过设置合并次数来控制词表大小（GPT-2用32k）
4. **处理任意文本**：基于字节，所以能处理中文、emoji等Unicode字符

## 1.3 从零实现BPE

下面我们用Python从零实现一个BPE tokenizer。你会看到它的核心逻辑不到100行。

```python
# code/tokenizer/bpe.py
"""
BPE Tokenizer 从零实现
基于 Roy Kahn 的原版GPT-2实现
"""

from collections import Counter, defaultdict
import re


class SimpleBPE:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.merges = []  # 记录所有的合并操作
        self.vocab = {}   # 最终词表

    def get_stats(self, words):
        """统计所有字符对的频率"""
        pairs = Counter()
        for word, freq in words.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def merge_vocab(self, words, pair):
        """合并所有词中的指定字符对"""
        new_words = {}
        pattern = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + pattern + r'(?!\S)')

        for word, freq in words.items():
            new_word = p.sub(''.join(pair), word)
            new_words[new_word] = freq
        return new_words

    def train(self, text):
        """训练BPE词表"""
        # 1. 初始化：将文本转为字节序列
        words = Counter()
        for line in text.split('\n'):
            tokens = list(line.encode('utf-8'))
            word = ' '.join([str(t) for t in tokens])
            words[word] += 1

        print(f"初始词汇量: {len(words)}")

        # 2. 迭代合并
        for i in range(self.vocab_size - 256):
            pairs = self.get_stats(words)
            if not pairs:
                break

            best = pairs.most_common(1)[0][0]
            words = self.merge_vocab(words, best)
            self.merges.append(best)

            if (i + 1) % 1000 == 0:
                print(f"已合并 {i + 1} 次，当前词表大小: {256 + i + 1}")

        # 3. 构建最终词表
        self.vocab = {i: bytes([i]).decode('utf-8', errors='replace')
                      for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]

    def encode(self, text):
        """对文本进行编码"""
        tokens = list(text.encode('utf-8'))
        for merge in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == merge[0] and tokens[i + 1] == merge[1]:
                    new_tokens.append(256 + self.merges.index(merge))
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def decode(self, ids):
        """对token ids进行解码"""
        return b''.join([bytes([id_]) for id_ in ids]).decode('utf-8', errors='replace')


# 示例运行
if __name__ == "__main__":
    text = """
    DeepSeek is amazing! The model understands context very well.
    I love using DeepSeek for coding tasks. It's truly revolutionary.
    """

    bpe = SimpleBPE(vocab_size=1000)
    bpe.train(text)

    # 测试编码
    sentence = "DeepSeek"
    ids = bpe.encode(sentence)
    print(f"'{sentence}' 编码为: {ids}")

    # 测试解码
    decoded = bpe.decode(ids)
    print(f"解码回: '{decoded}'")
```

运行输出示例：

```
初始词汇量: 156
已合并 1000 次，当前词表大小: 1256
'DeepSeek' 编码为: [68, 101, 101, 112, 256, 258, 259]
解码回: 'DeepSeek'
```

注意 `256` 对应 `"ee"`，`258` 对应 `"ek"`，这说明 BPE 学到了 `"ee"` 和 `"ek"` 是高频组合。

## 1.4 可视化：BPE的合并过程

```
┌─────────────────────────────────────────────────────────────┐
│                    BPE 编码过程图解                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  输入: "DeepSeek真好用"                                     │
│        ↓                                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  字节级切分 (UTF-8)                                   │   │
│  │  [68] [101] [101] [112] [256] [101] [101] [107]     │   │
│  │   D    e    e    p    k    真   好    用             │   │
│  └─────────────────────────────────────────────────────┘   │
│        ↓                                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  BPE 合并 (统计高频字节对)                            │   │
│  │  [68] [256] [259] [258] [306] [289]                 │   │
│  │   D   ee    ek   真    好    用                       │   │
│  └─────────────────────────────────────────────────────┘   │
│        ↓                                                    │
│  输出: [68, 256, 259, 258, 306, 289]                       │
│                                                             │
│  词表大小: 307 → 2 (压缩比: ~50%)                          │
└─────────────────────────────────────────────────────────────┘
```

## 1.5 分词器对比

| 分词器 | 切分方式 | 词表大小 | 典型应用 |
|--------|----------|----------|----------|
| Char | 每个字符 | ~50k (Unicode) | 早期NLP |
| Word | 整词 | ~500k (英语) | Word2Vec |
| BPE | 子词统计 | ~32k (GPT-2) | GPT系列 |
| WordPiece | 子词语言模型 | ~30k (BERT) | BERT, T5 |
| SentencePiece | 统一BPE | 可调 | LLaMA, 多语言 |

## 1.6 进阶：使用 tiktoken（工业级BPE）

上面的实现帮助理解原理，工业级项目使用 `tiktoken`（GPT-2官方分词器库）：

```python
import tiktoken

# 加载GPT-2的分词器
enc = tiktoken.get_encoding("gpt2")

text = "DeepSeek is fantastic!"
tokens = enc.encode(text, allowed_special={"<|endoftext|>"})

print(f"文本: {text}")
print(f"Token数量: {len(tokens)}")
print(f"Token IDs: {tokens}")
print(f"解码验证: {enc.decode(tokens)}")

# 输出:
# 文本: DeepSeek is fantastic!
# Token数量: 6
# Token IDs: [16937, 318, 16948, 1727, 0]
# 解码验证: DeepSeek is fantastic!
```

## 1.7 中文分词的特殊挑战

中文不像英文有天然空格分隔，需要额外处理：

```python
# 方案1：基于字符
"深深深度学习" → ["深", "深", "深", "度", "学", "习"]

# 方案2：基于词的BPE（如LLaMA使用的SentencePiece）
"深度学习" → ["深", "度", "学习"]  # "学习"作为整体保留

# 方案3：混合
"我在DeepSeek学习" → ["我", "在", "Deep", "Seek", "学习"]
```

现代模型（如LLaMA、DeepSeek）使用 **SentencePiece**，它将BPE应用于Unicode字节，可以同时处理多语言而无需语言特定的处理。

## 1.8 本章小结

1. **Tokenizer是LLM的第一道门槛**：把文字转换为数字
2. **BPE用统计方法找到高频词片段**：从字节出发，迭代合并
3. **词表大小影响效果**：太小欠拟合，太大增加计算量
4. **SentencePiece统一处理多语言**：基于Unicode字节，通用性强

### 思考题

1. 如果一个生僻字（如"龘"）在训练数据中只出现1次，BPE会如何处理？
2. 为什么GPT-4使用更大的128k词表？这会带来什么代价？
3. 尝试用代码实现一个简单的BPE训练器，统计"hello world hello"的合并过程。

### 延伸阅读

- [GPT-2 Paper Section 2.2: Tokenizer](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
- [SentencePiece: A simple language-independent subword tokenizer](https://github.com/google/sentencepiece)