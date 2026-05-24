# 第十二章：推理优化 — FlashAttention、PagedAttention与连续批处理

## 故事开场：速度的瓶颈

Bob训练了一个7B参数的模型，兴冲冲地部署上线。但他发现：

```
单次生成请求 (100 tokens):
- 延迟: 2.5 秒
- 吞吐量: 0.4 req/s
- 显存占用: 14GB

并发 10 个请求:
- 延迟: 15 秒
- 吞吐量: 0.7 req/s
- 显存溢出！
```

"为什么延迟这么高？"Bob问。

Alice解释："这是因为**推理的计算和显存管理没有优化**。"

## 12.1 推理的挑战

### 标准Transformer推理的问题

```
前向传播:
Token 1 → [Layer 1] → [Layer 2] → ... → [Layer N] → Token 2
Token 1,2 → [Layer 1] → ... → Token 3
...
```

每次生成一个新token，都需要**重新计算所有token的注意力**。

### 显存问题

```
KV Cache:
每个token在每一层都需要存储 K 和 V 向量

7B模型，50层，5120隐藏维度，FP16:
每层每token: 2 × 5120 × 2 bytes = 20KB
50层: 1MB/token
1000 tokens: 1GB显存只是KV Cache！

如果同时处理多个请求，显存会爆炸。
```

## 12.2 FlashAttention

### 核心思想

标准Attention的计算：

```
1. 计算 QKᵀ (需要 O(n²) 显存存储中间结果)
2. Softmax
3. 乘以 V

问题: n² 的中间矩阵需要大量显存
```

FlashAttention的核心思想：

> **分块计算，避免物化大的中间矩阵**

```
标准:                          FlashAttention:
QKᵀ (n×n) → softmax →         分成小块
                              block_1: Q1K1ᵀ → softmax → Q1V1
中间矩阵 O(n²) 太大！          block_2: Q2K2ᵀ → softmax → Q2V2
                              ...（流式处理）
                              最后合并
```

### 实现

```python
# code/inference/flash_attention.py
"""
FlashAttention 实现（简化版）
"""

import torch
import torch.nn.functional as F

def flash_attention(Q, K, V, block_size=128):
    """
    FlashAttention 分块实现

    参数:
        Q: (batch, seq_len_q, d_k)
        K: (batch, seq_len_k, d_k)
        V: (batch, seq_len_k, d_v)
        block_size: 每块的大小
    """
    batch_size, seq_len_q, d_k = Q.shape
    seq_len_k = K.shape[1]

    # 缩放因子
    scale = 1.0 / (d_k ** 0.5)

    # 输出
    output = torch.zeros_like(Q)

    # 分块计算
    for i in range(0, seq_len_q, block_size):
        # 当前块的query
        Q_block = Q[:, i:i+block_size, :]

        # 初始化当前块的累加器
        max_score = torch.full((batch_size, block_size, 1), -float('inf'), device=Q.device)
        exp_sum = torch.zeros(batch_size, block_size, 1, device=Q.device)
        result = torch.zeros(batch_size, block_size, d_k, device=Q.device)

        for j in range(0, seq_len_k, block_size):
            # 当前块的K和V
            K_block = K[:, j:j+block_size, :]
            V_block = V[:, j:j+block_size, :]

            # 计算注意力分数
            # (batch, block_q, d_k) @ (batch, d_k, block_k) -> (batch, block_q, block_k)
            scores = torch.einsum('bqd,bkd->bqk', Q_block, K_block) * scale

            # 数值稳定的softmax
            # scores: (batch, block_q, block_k)
            block_max = scores.max(dim=-1, keepdim=True)[0]
            block_max = torch.max(max_score[:, :block_size, :], block_max)

            # 指数和
            exp_scores = torch.exp(scores - block_max)
            exp_sum_block = exp_scores.sum(dim=-1, keepdim=True)

            # 累加到全局
            exp_sum_new = exp_sum + exp_sum_block
            result = (result * (exp_sum / exp_sum_new) + exp_scores / exp_sum_new) @ V_block

            max_score[:, :block_size, :] = block_max
            exp_sum = exp_sum_new

        output[:, i:i+block_size, :] = result

    return output


# PyTorch内置的scaled_dot_product_attention（FlashAttention实现）
def modern_flash_attention(Q, K, V, is_causal=True):
    """
    使用PyTorch 2.0+的FlashAttention

    等价于FlashAttention-2
    """
    # enable_flash_attention 是自动检测的
    output = F.scaled_dot_product_attention(
        Q, K, V,
        attn_mask=None,
        is_causal=is_causal  # 自动应用causal mask
    )
    return output
```

### 速度对比

```
序列长度: 2048, batch: 1

标准Attention:    2.5秒, 显存 8GB
FlashAttention:   0.3秒, 显存 0.5GB

加速: ~8×
显存节省: ~16×
```

## 12.3 PagedAttention

### 问题：KV Cache碎片化

```
传统方式处理多个序列:
Seq1: [Token1, Token2, ..., Token_n] → KV Cache
Seq2: [Token1, Token2, ..., Token_m] → KV Cache
...

问题: 每个序列预分配固定大小的KV Cache
      可能造成大量浪费（大多数序列不会用满）
```

### PagedAttention的解决

```
核心思想: 像操作系统的分页一样管理KV Cache

逻辑块 → 物理块（不一定连续）
KV Cache存储在物理内存中，按需分配
```

```python
# 简化版PagedAttention
class BlockManager:
    """块管理器"""
    def __init__(self, block_size=16, num_blocks=100):
        self.block_size = block_size
        self.blocks = [None] * num_blocks
        self.free_blocks = list(range(num_blocks))

    def allocate(self, num_tokens):
        """分配num_tokens需要的块"""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        if len(self.free_blocks) < num_blocks_needed:
            return None  # 内存不足

        allocated = []
        for _ in range(num_blocks_needed):
            block_id = self.free_blocks.pop()
            self.blocks[block_id] = torch.zeros(self.block_size, self.d_model)
            allocated.append(block_id)

        return allocated

    def free(self, block_ids):
        """释放块"""
        for block_id in block_ids:
            self.blocks[block_id] = None
            self.free_blocks.append(block_id)


def paged_attention(query, block_ids, block_manager):
    """PagedAttention计算"""
    # 收集所有相关的物理块
    k_blocks = [block_manager.blocks[bid]['k'] for bid in block_ids]
    v_blocks = [block_manager.blocks[bid]['v'] for bid in block_ids]

    # 拼接（逻辑上连续，物理上可能不连续）
    K = torch.cat(k_blocks, dim=0)
    V = torch.cat(v_blocks, dim=0)

    # 计算注意力
    return flash_attention(query, K, V)
```

### vLLM中的应用

```python
# 使用vLLM进行高效推理
from vllm import LLM, SamplingParams

# 初始化
llm = LLM(model="meta-llama/Llama-2-7b")

# 采样参数
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100
)

# 批量请求
outputs = llm.generate([
    "Write a story about:",
    "Explain quantum computing:",
    "Write Python code for:",
], sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

## 12.4 连续批处理（Continuous Batching）

### 问题：静态批处理

```
静态批处理:
Batch [Req1, Req2, Req3] (相同长度)

问题:
- Req1 完成后，GPU空闲等待
- 其他请求不能立即加入batch
- 资源利用率低
```

### 连续批处理

```
连续批处理:
1. 初始batch: [Req1, Req2]
2. Req2 完成，释放
3. 新请求 Req3 加入batch（动态）
4. Req1 继续（KV cache保留）

重叠执行，最大化GPU利用率
```

```python
# 连续批处理示意
class ContinuousBatching:
    def __init__(self, model, max_batch_size=8):
        self.model = model
        self.max_batch_size = max_batch_size
        self.active_requests = []

    def step(self):
        """每个step的处理"""
        # 1. 生成下一个token
        for req in self.active_requests:
            req.next_token = self.model.generate_step(req.input_ids)

        # 2. 检查完成
        completed = [req for req in self.active_requests if req.is_finished()]
        for req in completed:
            self.output.append(req.result)
            self.active_requests.remove(req)

        # 3. 添加新请求（如果有空槽）
        while len(self.active_requests) < self.max_batch_size and self.pending_requests:
            new_req = self.pending_requests.pop(0)
            self.active_requests.append(new_req)

        return completed
```

## 12.5 推理优化技术汇总

| 技术 | 解决的问题 | 效果 |
|------|----------|------|
| FlashAttention | 中间矩阵显存过大 | 显存↓，速度↑ |
| PagedAttention | KV Cache碎片化 | 显存利用率↑，吞吐量↑ |
| 连续批处理 | 静态批处理资源浪费 | GPU利用率↑，延迟稳定 |
| 量化 | 计算和存储成本高 | 显存↓，速度↑ |
| Speculative Decoding | 自回归生成慢 | 2-4× 加速 |
| 投机解码 | 自回归生成慢 | 2-4× 加速 |

## 12.6 Speculative Decoding

### 核心思想

自回归生成是串行的，每个token必须等前一个生成。

**投机解码**：用一个小模型（draft model）快速生成多个候选token，然后用大模型验证。

```
Draft Model (小): 生成 [token1, token2, token3, token4] (快速)
                 ↓
大模型验证:     [✓, ✓, ✗, ...]  (一次验证多个)
                 ↓
接受: token1, token2
拒绝: token3 (用大模型重新生成)
```

```python
def speculative_decode(draft_model, target_model, prompt, num_speculative=4):
    """
    投机解码

    参数:
        draft_model: 小模型，用于快速生成
        target_model: 大模型，用于验证
        prompt: 输入
        num_speculative: 候选token数量
    """
    # 1. Draft模型生成候选
    draft_tokens = []
    input_ids = tokenize(prompt)

    for _ in range(num_speculative):
        logits = draft_model(input_ids)
        next_token = torch.argmax(logits[:, -1, :])
        draft_tokens.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    # 2. Target模型批量验证
    target_logits = target_model(input_ids)

    # 3. 比较和接受
    accepted = 0
    for i, token in enumerate(draft_tokens):
        target_prob = F.softmax(target_logits[0, -(len(draft_tokens)-i), :], dim=-1)
        draft_prob = F.softmax(draft_model_logits[0, -(len(draft_tokens)-i), :], dim=-1)

        # 如果token在target模型中也高概率，接受
        if target_prob[token] >= draft_prob[token]:
            accepted += 1
        else:
            break  # 从这个位置开始用target模型

    return draft_tokens[:accepted]
```

## 12.7 推理优化实践

### 使用vLLM部署

```bash
# 安装vLLM
pip install vllm

# 启动服务
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b \
    --gpu-memory-utilization 0.9 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 256

# API调用
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-2-7b",
        "messages": [{"role": "user", "content": "Hello!"}]
    }'
```

### 使用Ray Serve部署

```python
# code/inference/deploy.py
"""
使用Ray Serve部署优化推理服务
"""

from ray import serve
from vllm import LLM

@serve.deployment(num_replicas=2, ray_actor_options={"num_gpus": 1})
class LLMDeployment:
    def __init__(self, model_name):
        self.llm = LLM(model=model_name)

    def generate(self, prompt, **kwargs):
        outputs = self.llm.generate([prompt], **kwargs)
        return outputs[0].outputs[0].text

# 部署
deploy = LLMDeployment.bind("meta-llama/Llama-2-7b")
serve.run(deploy)
```

## 12.8 本章小结

1. **FlashAttention**：分块计算避免O(n²)显存，支持长序列
2. **PagedAttention**：虚拟内存思想管理KV Cache，提高利用率
3. **连续批处理**：动态batch，最大化GPU利用率
4. **Speculative Decoding**：小模型预测，大模型验证，2-4×加速
5. **工具**：vLLM、TensorRT-LLM等已实现这些优化

### 思考题

1. FlashAttention相比标准Attention，在数学上是否等价？还是近似？
2. PagedAttention的block_size如何选择？太大或太小有什么影响？
3. 为什么投机解码能够加速？它的加速比受什么因素限制？

### 延伸阅读

- [FlashAttention: Fast and Memory-Efficient Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [PagedAttention](https://arxiv.org/abs/2309.06119)
- [vLLM: Easy, Fast, and Cheap LLM Serving](https://vllm.ai/)
- [Speculative Decoding](https://arxiv.org/abs/2211.17192)