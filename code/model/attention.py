"""
Multi-Head Attention 实现
包含 FlashAttention 的优化版本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """标准 Multi-Head Attention"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.d_k)

        # 可学习的投影矩阵
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """将 embedding 分割成多个头"""
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, num_heads, seq_len, d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数:
            query: (batch, seq_len_q, d_model)
            key: (batch, seq_len_k, d_model)
            value: (batch, seq_len_v, d_model)
            mask: 可选，注意力掩码

        返回:
            output: (batch, seq_len_q, d_model)
            attention_weights: (batch, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = query.size(0)

        # 1. 线性投影
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # 2. 分割成多个头
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 3. 计算注意力
        # QK^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 4. 加权求和
        context = torch.matmul(attention_weights, V)
        # context: (batch, num_heads, seq_len, d_k)

        # 5. 合并多头
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.d_model)

        # 6. 最终投影
        output = self.W_o(context)

        return output, attention_weights


class FlashAttention(nn.Module):
    """FlashAttention 实现"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = dropout

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool = True,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用 PyTorch 内置的 scaled_dot_product_attention（FlashAttention 实现）

        参数:
            query: (batch, seq_len_q, d_model)
            key: (batch, seq_len_k, d_model)
            value: (batch, seq_len_v, d_model)
            is_causal: 是否应用 causal mask
            mask: 可选的自定义掩码，True=保留，False=屏蔽

        返回:
            output: (batch, seq_len_q, d_model)
            attention_weights: (batch, num_heads, seq_len_q, seq_len_k)
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.size(1)

        # 线性投影
        Q = self.W_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)

        # 转换为 SDPA 期望的加性掩码 (0=保留, -inf=屏蔽)
        additive_mask = None
        if mask is not None:
            additive_mask = torch.zeros_like(mask, dtype=Q.dtype)
            additive_mask[~mask] = float('-inf')

        # 使用 SDPA (FlashAttention)
        output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=additive_mask,
            is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0
        )

        # 合并头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        output = self.W_o(output)

        # 近似返回 weights（用于可视化）
        with torch.no_grad():
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            if is_causal:
                causal_mask = torch.triu(
                    torch.ones(seq_len_q, seq_len_k, device=query.device),
                    diagonal=1
                ).bool()
                scores = scores.masked_fill(causal_mask, float('-inf'))
            if mask is not None:
                scores = scores.masked_fill(~mask, float('-inf'))
            weights = F.softmax(scores, dim=-1)

        return output, weights


class RoPEAttention(nn.Module):
    """带 RoPE (Rotary Position Encoding) 的 Attention"""

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # RoPE 频率预计算
        self.register_buffer('freqs', self._precompute_freqs(max_seq_len, theta))

    def _precompute_freqs(self, max_seq_len: int, theta: float) -> torch.Tensor:
        """预计算 RoPE 频率"""
        freqs = 1.0 / (theta ** (torch.arange(0, self.d_k, 2).float() / self.d_k))
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, freqs)
        freqs = torch.polar(torch.ones_like(freqs), freqs)  # 转为复数形式
        return freqs

    def apply_rope(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """应用旋转位置编码"""
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rot = torch.view_as_real(x_complex * freqs.unsqueeze(0))
        return x_rot.flatten(-2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        is_causal: bool = True
    ) -> torch.Tensor:
        """
        参数:
            query: (batch, seq_len_q, d_model)
            key: (batch, seq_len_k, d_model)
            value: (batch, seq_len_v, d_model)
            position_ids: 位置索引，用于获取 RoPE 频率，默认自动创建
            is_causal: 是否应用 causal mask
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.size(1)

        # 投影
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # 分割头
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)

        # 应用 RoPE（支持交叉注意力：Q和K可以有不同的位置编码）
        if position_ids is None:
            position_ids_q = torch.arange(seq_len_q, device=query.device)
            position_ids_k = torch.arange(seq_len_k, device=key.device)
        else:
            position_ids_q = position_ids
            position_ids_k = position_ids

        freqs_q = self.freqs[position_ids_q]
        freqs_k = self.freqs[position_ids_k]
        Q = self.apply_rope(Q, freqs_q)
        K = self.apply_rope(K, freqs_k)

        # FlashAttention 计算
        output = F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal)

        # 合并头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        output = self.W_o(output)

        return output


if __name__ == "__main__":
    # 测试
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8

    # 随机输入
    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)

    # 标准 Attention
    mha = MultiHeadAttention(d_model, num_heads)
    out1, _ = mha(q, k, v)
    print(f"MultiHeadAttention output shape: {out1.shape}")

    # FlashAttention
    flash = FlashAttention(d_model, num_heads)
    out2, _ = flash(q, k, v, is_causal=True)
    print(f"FlashAttention output shape: {out2.shape}")

    # RoPE Attention
    rope_attn = RoPEAttention(d_model, num_heads)
    out3 = rope_attn(q, k, v)
    print(f"RoPEAttention output shape: {out3.shape}")