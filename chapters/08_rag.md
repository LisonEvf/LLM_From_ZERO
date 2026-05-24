# 第八章：知识的扩展 — RAG检索增强生成

## 故事开场：静态知识的困境

Alice用GPT-4构建了一个智能客服机器人。客户问：

```
用户: "你们公司今天有什么新产品？"
Bot: "抱歉，我的知识截止到2023年12月，无法回答关于今天的问题。"
```

"但我们明明有产品数据库啊！"Alice灵机一动，"为什么不能让模型**实时检索**最新信息？"

这就是 **RAG（Retrieval-Augmented Generation）** 的核心思想：**让模型能够访问外部知识**。

## 8.1 RAG概述

### 什么是RAG？

RAG = Retrieval（检索）+ Augmented（增强）+ Generation（生成）

```
传统LLM:
User → [LLM] → Response (受限于训练数据)

RAG:
User → Query → [检索器] → 相关文档
                ↓
         [LLM + 文档] → Response (包含最新信息)
```

### 为什么需要RAG？

1. **知识时效性**：模型知识有截止日期
2. **知识覆盖**：模型不可能记住所有细节
3. **可解释性**：可以引用来源
4. **成本**：比微调更便宜、更快

## 8.2 RAG的核心问题

### 问题1：检索什么？

**向量数据库（Vector Database）**

```
文档 → 分块 → 嵌入 → 存储到向量数据库

用户查询 → 嵌入 → 在向量数据库中搜索相似块
```

**嵌入模型选择**：
- 英文：sentence-transformers/all-MiniLM-L6-v2
- 中文：moka-ai/m3e-base, shibing624/text2vec-base-chinese

### 问题2：如何检索？

两种主要范式：

```
1. 稀疏检索 (Sparse Retrieval / BM25)
   - 使用TF-IDF
   - 基于词频统计
   - 缺点：不理解语义

2. 密集检索 (Dense Retrieval)
   - 使用神经网络嵌入
   - 理解语义
   - 缺点：需要更多计算
```

### 问题3：如何融合？

**混合检索（Hybrid Retrieval）**

```python
def hybrid_retrieval(query, vector_db, bm25_index, k=10):
    """
    混合检索：结合稀疏和密集
    """
    # 密集检索：向量相似度
    dense_results = vector_db.search(query, k=k)

    # 稀疏检索：BM25
    sparse_results = bm25_index.search(query, k=k)

    # 融合：倒数排名融合 (RRF)
    fused_scores = {}
    for i, doc in enumerate(dense_results):
        doc_id = doc.id
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + i)

    for i, doc in enumerate(sparse_results):
        doc_id = doc.id
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + i)

    # 排序
    sorted_docs = sorted(fused_scores.items(), key=lambda x: -x[1])[:k]
    return sorted_docs
```

## 8.3 RAG流程详解

```
┌─────────────────────────────────────────────────────────────┐
│                      RAG 完整流程                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  阶段1：索引 (Indexing)                                     │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   │
│  │ 文档    │ → │ 切分    │ → │ 嵌入    │ → │ 存储    │   │
│  │         │   │ (chunk) │   │ (vector)│   │(向量DB) │   │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘   │
│                                                             │
│  阶段2：检索 (Retrieval)                                    │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   │
│  │ 用户    │ → │ 嵌入    │ → │ 相似度  │ → │ Top-K   │   │
│  │ 查询    │   │ 查询    │   │ 计算    │   │ 检索    │   │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘   │
│                                                             │
│  阶段3：生成 (Generation)                                   │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐                   │
│  │ 查询+   │ → │ Prompt  │ → │ LLM     │ → Response       │
│  │ 检索结果│   │ 组装    │   │ 生成    │                   │
│  └─────────┘   └─────────┘   └─────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

## 8.4 从零实现RAG

```python
# code/rag/rag_system.py
"""
RAG 系统实现
"""

from typing import List, Tuple
import numpy as np


class SimpleChunker:
    """简单的文本分块器"""
    def __init__(self, chunk_size=200, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:
        """将文本切分为块"""
        words = text.split()
        chunks = []

        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            start += self.chunk_size - self.overlap

        return chunks


class SimpleEmbedder:
    """简单的嵌入模型（用于演示）"""
    def __init__(self, dim=384):
        self.dim = dim
        # 随机初始化，实际应该用预训练模型
        self.weights = np.random.randn(10000, dim) * 0.02

    def embed(self, texts: List[str]) -> np.ndarray:
        """将文本嵌入为向量"""
        # 简化：使用hash模拟
        embeddings = []
        for text in texts:
            hash_val = hash(text) % 10000
            embeddings.append(self.weights[hash_val])

        return np.array(embeddings)


class VectorStore:
    """简单的向量数据库"""
    def __init__(self, dim=384):
        self.dim = dim
        self.vectors = []
        self.metadata = []

    def add(self, vectors: np.ndarray, metadata: List[dict]):
        """添加向量和元数据"""
        self.vectors.extend(vectors)
        self.metadata.extend(metadata)

    def search(self, query_vector: np.ndarray, k=5) -> List[Tuple[str, float]]:
        """搜索最相似的k个向量"""
        scores = []
        for vec in self.vectors:
            sim = self.cosine_sim(query_vector, vec)
            scores.append(sim)

        # 取top-k
        top_k_idx = np.argsort(scores)[-k:][::-1]

        results = []
        for idx in top_k_idx:
            results.append((self.metadata[idx]['text'], scores[idx]))

        return results

    def cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


class RAGSystem:
    """RAG系统"""
    def __init__(self):
        self.chunker = SimpleChunker(chunk_size=100, overlap=20)
        self.embedder = SimpleEmbedder()
        self.vector_store = VectorStore()

    def index_documents(self, documents: List[str]):
        """索引文档"""
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk(doc)
            all_chunks.extend(chunks)

        # 嵌入
        embeddings = self.embedder.embed(all_chunks)

        # 存储
        metadata = [{'text': chunk} for chunk in all_chunks]
        self.vector_store.add(embeddings, metadata)

        print(f"索引完成: {len(all_chunks)} 个块")

    def retrieve(self, query: str, k=5) -> List[str]:
        """检索相关文档"""
        query_emb = self.embedder.embed([query])
        results = self.vector_store.search(query_emb[0], k=k)
        return [text for text, score in results]

    def answer(self, query: str, context: str) -> str:
        """生成回答（这里用简单拼接作为演示）"""
        prompt = f"""基于以下信息回答问题。如果信息不足，说"我没有足够信息回答这个问题。"

问题: {query}

参考信息:
{context}

回答:"""

        # 实际应用中，这里应该调用LLM
        return f"基于检索到的信息回答: {query}"


# 示例使用
def main():
    # 初始化
    rag = RAGSystem()

    # 索引文档
    documents = [
        "DeepSeek是一家中国AI公司，专注于大语言模型的研发。",
        "DeepSeek-V2是他们最新的开源模型，在多项基准测试中表现优异。",
        "DeepSeek支持开源，代码和模型权重都可以在GitHub上获取。",
        "公司的产品包括DeepSeek-Coder、DeepSeek-Math等专业化模型。"
    ]

    rag.index_documents(documents)

    # 检索
    query = "DeepSeek是什么公司？"
    retrieved = rag.retrieve(query, k=2)

    print(f"\n查询: {query}")
    print(f"检索结果: {retrieved}")

    # 生成
    context = "\n".join(retrieved)
    answer = rag.answer(query, context)
    print(f"回答: {answer}")


if __name__ == "__main__":
    main()
```

## 8.5 RAG评估

### 检索评估指标

```python
def evaluate_retrieval(retrieved_docs, relevant_docs):
    """评估检索质量"""
    # Precision@K: 检索结果中有多少相关
    for k in [1, 3, 5]:
        precision = len(set(retrieved_docs[:k]) & set(relevant_docs)) / k

    # MRR: 第一个相关文档的排名倒数均值
    for i, doc in enumerate(retrieved_docs):
        if doc in relevant_docs:
            mrr = 1 / (i + 1)
            break

    return {"precision@k": precision, "mrr": mrr}
```

### 生成评估

| 指标 | 说明 |
|------|------|
| 答案正确性 | 答案是否准确 |
| 答案相关性 | 答案是否针对问题 |
| 引用准确性 | 引用的文档是否正确 |
| 上下文利用率 | 是否有效利用检索到的信息 |

### RAGAS

RAGAS（RAG Assessment）是一种综合评估方法：

```python
# RAGAS评分
# faithfulness: 答案是否忠实于检索到的上下文
# answer_relevancy: 答案是否针对问题
# context_relevancy: 检索到的上下文是否相关
```

## 8.6 RAG的进阶技术

### 1. 句子窗口检索

```python
# 不仅检索相关块，还检索周围的块
def sentence_window_retrieval(query, window_size=3):
    # 先检索
    initial_results = vector_db.search(query, k=1)

    # 获取窗口块
    full_context = []
    for chunk in initial_results:
        # 获取前后的块
        prev_chunks = get_previous_chunks(chunk, window_size)
        next_chunks = get_next_chunks(chunk, window_size)
        full_context.extend(prev_chunks + [chunk] + next_chunks)

    return full_context
```

### 2. 迭代式RAG

```python
def iterative_rag(query, max_iterations=3):
    """多次检索，逐步细化"""
    current_context = ""
    current_query = query

    for i in range(max_iterations):
        # 检索
        results = retrieve(current_query, k=5)

        # 评估是否足够
        if is_context_sufficient(results):
            break

        # 细化查询
        current_query = refine_query(current_query, results)
        current_context += results

    return current_context
```

### 3. 使用LangChain实现RAG

```python
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

# 加载文档
loader = WebBaseLoader(["https://deepseek.com/about"])
documents = loader.load()

# 切分
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# 向量化
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings)

# 构建RAG链
qa_chain = RetrievalQA.from_chain_type(
    llm=HuggingFacePipeline(...),
    retriever=vectorstore.as_retriever(),
    chain_type="stuff"
)

# 查询
result = qa_chain.run("DeepSeek是什么时候成立的？")
print(result)
```

## 8.7 本章小结

1. **RAG目的**：让LLM访问外部最新知识
2. **核心组件**：检索器 + 生成器
3. **检索方式**：稀疏（BM25）、密集（向量）、混合（RRF）
4. **评估指标**：Precision@K, MRR, RAGAS
5. **进阶技术**：句子窗口、迭代检索、多跳推理

### 思考题

1. RAG和微调相比，各有什么优缺点？什么场景适合用RAG而不是微调？
2. 如果检索到的文档相互矛盾，模型应该如何处理？
3. 如何防止RAG系统被恶意注入（检索到有害内容）？

### 延伸阅读

- [RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [LangChain Documentation](https://python.langchain.com/)
- [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15254)