"""
GPT 模型实现
包含完整的语言模型训练和推理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .transformer_block import TransformerBlock, MoETransformerBlock
from .attention import RoPEAttention


class GPTEmbeddings(nn.Module):
    """Token Embedding + Position Embedding"""

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int = 2048):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None):
        """
        参数:
            input_ids: (batch, seq_len)
            position_ids: 可选，默认自动创建
        """
        seq_len = input_ids.size(1)

        token_emb = self.token_embedding(input_ids)

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device)

        pos_emb = self.position_embedding(position_ids)

        return token_emb + pos_emb


class GPTModel(nn.Module):
    """完整的 GPT 模型"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.embeddings = GPTEmbeddings(vocab_size, d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 权重绑定：token embedding 和 lm head 共享
        self.lm_head.weight = self.embeddings.token_embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        参数:
            input_ids: (batch, seq_len)
            attention_mask: 可选
            position_ids: 可选

        返回:
            logits: (batch, seq_len, vocab_size)
        """
        # Embedding
        x = self.embeddings(input_ids, position_ids)
        x = self.dropout(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask=attention_mask)

        x = self.norm(x)

        # 输出 logits
        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        自回归生成

        参数:
            input_ids: (batch, seq_len) 初始输入
            max_new_tokens: 生成的最大 token 数
            temperature: 温度采样，>1 更随机，<1 更确定
            top_k: 只保留 top_k 个 token
            top_p: nucleus sampling

        返回:
            generated: (batch, seq_len + max_new_tokens)
        """
        self.eval()

        for _ in range(max_new_tokens):
            # 截断到最大长度
            input_ids_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]

            # 前向传播
            logits = self.forward(input_ids_cond)

            # 取最后一个位置的 logits
            logits = logits[:, -1, :] / temperature

            # Top-p (nucleus) 过滤 - 先应用
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Top-k 过滤 - 后应用
            if top_k is not None:
                topk_logits, topk_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(1, topk_indices, topk_logits)

            # 采样
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 拼接
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


class RoPEGPTModel(nn.Module):
    """使用 RoPE 的 GPT 模型（类似 LLaMA）"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

        if use_moe:
            self.blocks = nn.ModuleList([
                MoETransformerBlock(d_model, num_heads, d_ff, num_experts, top_k, dropout)
                for _ in range(num_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                TransformerBlock(d_model, num_heads, d_ff, dropout, activation="silu")
                for _ in range(num_layers)
            ])

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # RoPE 位置编码在 attention 层内部处理

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, device=input_ids.device)

        x = self.token_embedding(input_ids)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, mask=attention_mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100, **kwargs):
        """简化版生成"""
        self.eval()
        for _ in range(max_new_tokens):
            input_ids_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]
            logits = self.forward(input_ids_cond)[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


def create_gpt_config(model_size: str) -> dict:
    """创建预设大小的 GPT 配置"""
    configs = {
        "gpt2-small": {
            "vocab_size": 50257,
            "d_model": 768,
            "num_heads": 12,
            "num_layers": 12,
            "d_ff": 3072,
            "max_seq_len": 1024
        },
        "gpt2-medium": {
            "vocab_size": 50257,
            "d_model": 1024,
            "num_heads": 16,
            "num_layers": 24,
            "d_ff": 4096,
            "max_seq_len": 1024
        },
        "gpt2-large": {
            "vocab_size": 50257,
            "d_model": 1280,
            "num_heads": 20,
            "num_layers": 36,
            "d_ff": 5120,
            "max_seq_len": 1024
        },
        "gpt2-xl": {
            "vocab_size": 50257,
            "d_model": 1600,
            "num_heads": 25,
            "num_layers": 48,
            "d_ff": 6400,
            "max_seq_len": 1024
        }
    }
    return configs.get(model_size, configs["gpt2-small"])


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    config = create_gpt_config("gpt2-small")
    model = GPTModel(**config)

    # 统计参数量
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GPT-2 Small 参数量: {n_params / 1e6:.2f}M")

    # 测试前向传播
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))

    logits = model(input_ids)
    print(f"Output logits shape: {logits.shape}")  # (batch, seq_len, vocab_size)

    # 测试生成
    prompt = torch.randint(0, config["vocab_size"], (1, 5))
    generated = model.generate(prompt, max_new_tokens=10)
    print(f"Generated shape: {generated.shape}")  # (1, 15)