"""
Transformer Block 实现
包含标准 FFN 和 MoE 变体
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .attention import MultiHeadAttention, RoPEAttention


class FeedForward(nn.Module):
    """标准 FFN (Position-wise Feed-Forward Network)"""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: str = "gelu",
        dropout: float = 0.1
    ):
        super().__init__()

        if activation == "gelu":
            act = nn.GELU()
        elif activation == "silu":
            act = nn.SiLU()
        elif activation == "relu":
            act = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            act,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class TransformerBlock(nn.Module):
    """标准 Transformer Block (Pre-Norm)"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = FeedForward(d_model, d_ff, activation, dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-Norm: 先做 LayerNorm，再做 Attention
        attn_out, _ = self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + attn_out  # 残差连接

        # Pre-Norm: 先做 LayerNorm，再做 FFN
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out

        return x


class MoEFeedForward(nn.Module):
    """MoE (Mixture of Experts) FFN"""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int,
        activation: str = "gelu",
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 门控网络
        self.gate = nn.Linear(d_model, num_experts, bias=False)

        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU() if activation == "gelu" else nn.SiLU(),
                nn.Linear(d_ff, d_model)
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (batch, seq_len, d_model)
        返回:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        # 门控计算
        gate_logits = self.gate(x)  # (batch, seq_len, num_experts)
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        # 初始化输出
        output = torch.zeros_like(x)

        # 按专家分组，批量处理
        for expert_id, expert in enumerate(self.experts):
            # 找出哪些位置的 token 需要这个专家（每个 token 可能被多个专家处理）
            # top_k_indices: (batch, seq_len, top_k)
            # 找出所有用到这个专家的位置（对于每个 top_k 位置）
            for k_idx in range(self.top_k):
                mask_k = top_k_indices[:, :, k_idx] == expert_id  # (batch, seq_len)

                if mask_k.any():
                    # 收集这个专家处理的 token
                    active_tokens = x[mask_k]  # (num_active, d_model)
                    active_weights = top_k_weights[:, :, k_idx][mask_k]  # (num_active,)

                    # 批量计算
                    expert_output = expert(active_tokens)  # (num_active, d_model)

                    # 累加到对应位置
                    output[mask_k] += expert_output * active_weights.unsqueeze(-1)

        return output


class MoETransformerBlock(nn.Module):
    """使用 MoE FFN 的 Transformer Block"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_experts: int,
        top_k: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.moe = MoEFeedForward(d_model, d_ff, num_experts, top_k)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-Attention
        attn_out, _ = self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + attn_out

        # MoE FFN
        moe_out = self.moe(self.norm2(x))
        x = x + moe_out

        return x


if __name__ == "__main__":
    # 测试
    d_model = 512
    num_heads = 8
    d_ff = 2048

    # 标准 Block
    block = TransformerBlock(d_model, num_heads, d_ff)
    x = torch.randn(2, 10, d_model)
    out = block(x)
    print(f"TransformerBlock output: {out.shape}")

    # MoE Block
    moe_block = MoETransformerBlock(d_model, num_heads, d_ff, num_experts=8, top_k=2)
    out = moe_block(x)
    print(f"MoETransformerBlock output: {out.shape}")