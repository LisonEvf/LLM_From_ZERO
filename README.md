# 从零构建大语言模型：从GPT诞生到DeepSeek的技术之旅

<img src="assets/images/logo.png" width=150>

> "This is the story of how we taught machines to understand human language — one technical breakthrough at a time."

本项目以**叙事的方式**展开大语言模型的发展历程，每一个章节都是故事的一个篇章。从Tokenize的设计智慧讲起，一路蜿蜒至Attention的革命、GPT的Scaling、MoE的稀疏化、量化的压缩，最终抵达现代LLM的完整技术栈。

---

## 如何阅读本手册

本项目包含三种阅读路径：

1. **故事优先** → 从 `chapters/` 目录开始，按章节顺序阅读（适合入门）
2. **代码优先** → 从 `notebooks/` 目录开始，运行每个notebook（适合实践）
3. **深入研究** → 从 `code/` 目录导入模块，结合论文理解（适合进阶）

---

## 项目结构

```
LLM_From_ZERO/
├── chapters/              # 各章文字内容（Markdown）
│   ├── 01_tokenizer.md    # 第一章：Tokenizer设计哲学
│   ├── 02_embedding.md     # 第二章：词嵌入与位置编码
│   ├── 03_attention.md     # 第三章：Attention革命
│   ├── 04_gpt_scaling.md   # 第四章：GPT与Scaling Law
│   ├── 05_rope.md          # 第五章：RoPE旋转位置编码
│   ├── 06_moe.md           # 第六章：MoE专家混合
│   ├── 07_quantization.md  # 第七章：量化技术
│   ├── 08_rag.md           # 第八章：RAG检索增强
│   ├── 09_function_calling.md  # 第九章：工具调用
│   ├── 10_deepseek.md      # 第十章：DeepSeek集大成
│   ├── 11_finetuning.md    # 第十一章：微调方法
│   └── 12_inference.md     # 第十二章：推理优化
├── code/                  # 模块化Python代码
│   ├── tokenizer/          # BPE, WordPiece实现
│   ├── model/              # Attention, FeedForward, Transformer, GPT
│   ├── training/           # 预训练、微调（LoRA等）
│   └── inference/          # FlashAttention、量化演示
├── notebooks/             # 交互式Jupyter演示
│   ├── 01_BPE_Tokenizer.ipynb
│   ├── 02_Embeddings_Positions.ipynb
│   └── ...
├── docs/                  # 生成的手册（PDF/HTML）
├── assets/                # 图片、字体、样式
│   ├── drawio/            # 示意图源文件
│   └── images/            # 编译后的图片
├── README.md
├── requirements.txt
├── setup.sh              # 一键环境安装
├── CONTRIBUTING.md
└── LICENSE               # MIT
```

---

## 故事线 The Narrative

### 第一章：文本的数字化 — Tokenizer的设计哲学

为什么从Tokenizer开始？因为语言模型的第一道门槛是：**如何把人类的文字变成机器能认识的数字？**

字符级太低效，词级太死板。GPT-2带来的BPE（字节对编码）是一种优雅的妥协——它用统计的方法，自动找到高频的"词片段"，既不像字符那样啰嗦，也不像词那样局限。

[→ 开始阅读：chapters/01_tokenizer.md](chapters/01_tokenizer.md)
[→ 运行代码：notebooks/01_BPE_Tokenizer.ipynb](notebooks/01_BPE_Tokenizer.ipynb)

---

### 第二章：意义的表示 — 从词嵌入到上下文表示

Token只是ID，真正的含义在**嵌入（Embedding）**中。

2013年的Word2Vec开创了词向量的先河——"king - man + woman ≈ queen"。但这解决不了一词多义。Transformer带来的**自注意力机制**，让每个词的表示变成上下文的函数。

[→ 开始阅读：chapters/02_embedding.md](chapters/02_embedding.md)
[→ 运行代码：notebooks/02_Embeddings_Positions.ipynb](notebooks/02_Embeddings_Positions.ipynb)

---

### 第三章：Attention革命 — Transformer架构的诞生

2017年，Google的论文"Attention is All You Need"改变了AI的发展方向。

RNN的问题：**无法并行，处理长序列时会遗忘**。Transformer的解决方案：**用注意力机制捕捉任意位置的依赖关系**，同时支持并行训练。

[→ 开始阅读：chapters/03_attention.md](chapters/03_attention.md)
[→ 运行代码：notebooks/03_Sliding_Window_Attention.ipynb](notebooks/03_Sliding_Window_Attention.ipynb)

---

### 第四章：GPT的诞生 — 语言模型的Scaling Law

**GPT-1 (2018)**：微调即可完成多任务
**GPT-2 (2019)**：无需微调，prompt即可泛化
**GPT-3 (2020)**：1750亿参数，few-shot learning震惊业界

GPT-3证明了**Scaling Law**：模型越大，能力越强。但这也带来了问题——推理成本太高。

[→ 开始阅读：chapters/04_gpt_scaling.md](chapters/04_gpt_scaling.md)
[→ 运行代码：notebooks/06_GPT3_Model.ipynb](notebooks/06_GPT3_Model.ipynb)

---

### 第五章：让上下文更长 — RoPE旋转位置编码

Transformer的原始位置编码是**加性**的，GPT-2用的是可学习位置编码。但这有一个致命问题：**无法外推到训练长度以外**。

LLaMA/DeepSeek采用的**RoPE**用旋转矩阵编码位置信息，让模型能够处理超长上下文。

[→ 开始阅读：chapters/05_rope.md](chapters/05_rope.md)
[→ 运行代码：notebooks/07_RoPE_Implementation.ipynb](notebooks/07_RoPE_Implementation.ipynb)

---

### 第六章：高效之道 — MoE专家混合

如果每个Token都要经过所有参数，计算量是巨大的。**MoE（Mixture of Experts）**的思路是：让专家分工，节省计算。

[→ 开始阅读：chapters/06_moe.md](chapters/06_moe.md)
[→ 运行代码：notebooks/04_MoE_Architecture.ipynb](notebooks/04_MoE_Architecture.ipynb)

---

### 第七章：让大模型变小 — 量化技术

1750亿参数的模型，FP16也要350GB显存。推理成本高得离谱。

**量化**的核心思想：用更少的bit表示权重。

[→ 开始阅读：chapters/07_quantization.md](chapters/07_quantization.md)
[→ 运行代码：notebooks/05_Quantization.ipynb](notebooks/05_Quantization.ipynb)

---

### 第八章：知识的扩展 — RAG检索增强生成

大模型的知识是**静态的**（训练截止日期），无法即时更新。**RAG（Retrieval-Augmented Generation）**让模型能够检索外部知识。

[→ 开始阅读：chapters/08_rag.md](chapters/08_rag.md)
[→ 运行代码：notebooks/09_RAG_Retrieval-Augmented_Generation.ipynb](notebooks/09_RAG_Retrieval-Augmented_Generation.ipynb)

---

### 第九章：工具的力量 — Function Calling与Agent

大模型能生成文本，但无法执行动作。**Function Calling**让模型能够调用外部工具。

[→ 开始阅读：chapters/09_function_calling.md](chapters/09_function_calling.md)
[→ 运行代码：notebooks/10_Function_Calling.ipynb](notebooks/10_Function_Calling.ipynb)

---

### 第十章：汇聚一切 — DeepSeek风格完整模型

DeepSeek-V4代表了现代LLM技术的集大成者。

[→ 开始阅读：chapters/10_deepseek.md](chapters/10_deepseek.md)
[→ 运行代码：notebooks/08_DeepSeek_Integration.ipynb](notebooks/08_DeepSeek_Integration.ipynb)

---

### 第十一章：微调的艺术 — SFT、RLHF与LoRA

[→ 开始阅读：chapters/11_finetuning.md](chapters/11_finetuning.md)
[→ 运行代码：notebooks/11_Training_and_Inference_Optimization.ipynb](notebooks/11_Training_and_Inference_Optimization.ipynb)

---

### 第十二章：推理优化 — FlashAttention、PagedAttention与连续批处理

[→ 开始阅读：chapters/12_inference.md](chapters/12_inference.md)

---

## 快速开始

```bash
# 方式1：一键安装（推荐）
bash setup.sh

# 方式2：手动安装
pip install -r requirements.txt

# 运行第一个notebook
jupyter notebook notebooks/01_BPE_Tokenizer.ipynb
```

---

## 技术发展时间线 Timeline

```
2013  Word2Vec        词嵌入时代开启
2017  Transformer    Attention革命诞生
2018  GPT-1          生成式预训练探索
2019  GPT-2          BPE + Zero-shot突破
2020  GPT-3          175B参数，Few-shot震惊业界
2020  Longformer     Sliding Window Attention
2021  Switch Trans.  MoE稀疏化先驱
2022  Chinchilla     Scaling Law再思考
2023  LLaMA          开源LLM新纪元
2023  GPT-4          多模态 + Function Calling
2023  GPTQ/AWQ       量化技术成熟
2024  DeepSeek-V4    RoPE + MoE + 量化集大成
2024  FlashAttention3   硬件感知的注意力优化
```

---

## 参考资料

- **BPE/Tokenization** — [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf), [SentencePiece](https://github.com/google/sentencepiece)
- **Attention & Transformer** — [Attention is All You Need](https://arxiv.org/abs/1706.03762), [FlashAttention](https://arxiv.org/abs/2205.14135)
- **Position Encoding** — [RoFormer](https://arxiv.org/abs/2104.09864), [Llama](https://arxiv.org/abs/2302.13971)
- **MoE** — [Switch Transformer](https://arxiv.org/abs/2101.03961), [DeepSeek-MoE](https://arxiv.org/abs/2401.14166)
- **Quantization** — [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978)
- **Agent & Tool Use** — [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling), [ReAct](https://arxiv.org/abs/2210.03629)

---

## 贡献者 Contributors

[@Claude](https://github.com/MiniMax-AI) [@MiniMax](https://github.com/MiniMax-AI)

---

*本项目旨在以故事的方式，帮助开发者理解LLM技术的演进脉络，而非碎片化地罗列知识点。*