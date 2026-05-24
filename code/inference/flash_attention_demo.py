"""
FlashAttention 演示
对比标准 Attention 和 FlashAttention 的速度和显存
"""

import torch
import torch.nn.functional as F
import time
import math
from typing import Tuple


def standard_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    标准 Attention 实现

    参数:
        Q: (batch, num_heads, seq_len_q, d_k)
        K: (batch, num_heads, seq_len_k, d_k)
        V: (batch, num_heads, seq_len_v, d_v)
        mask: 可选掩码

    返回:
        output: (batch, num_heads, seq_len_q, d_v)
        attention_weights: (batch, num_heads, seq_len_q, seq_len_k)
    """
    d_k = Q.size(-1)

    # 计算注意力分数
    # (batch, num_heads, seq_len_q, d_k) @ (batch, num_heads, d_k, seq_len_k)
    # -> (batch, num_heads, seq_len_q, seq_len_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))

    # Softmax
    attention_weights = F.softmax(scores, dim=-1)

    # 加权求和
    output = torch.matmul(attention_weights, V)

    return output, attention_weights


def flash_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    is_causal: bool = True
) -> torch.Tensor:
    """
    使用 PyTorch 内置的 FlashAttention (SDPA)

    参数:
        Q: (batch, num_heads, seq_len_q, d_k)
        K: (batch, num_heads, seq_len_k, d_k)
        V: (batch, num_heads, seq_len_v, d_v)
        is_causal: 是否应用因果掩码

    返回:
        output: (batch, num_heads, seq_len_q, d_v)
    """
    # scaled_dot_product_attention 自动使用 FlashAttention
    output = F.scaled_dot_product_attention(
        Q, K, V,
        attn_mask=None,
        is_causal=is_causal,
        dropout_p=0.0
    )
    return output


def benchmark_attention(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    d_k: int,
    device: str = "cuda"
) -> dict:
    """
    基准测试：对比标准 Attention 和 FlashAttention

    参数:
        batch_size: batch 大小
        num_heads: 注意力头数
        seq_len: 序列长度
        d_k: 头维度
        device: "cuda" 或 "cpu"

    返回:
        包含时间和显存统计的字典
    """
    # 创建输入
    Q = torch.randn(batch_size, num_heads, seq_len, d_k, device=device)
    K = torch.randn(batch_size, num_heads, seq_len, d_k, device=device)
    V = torch.randn(batch_size, num_heads, seq_len, d_k, device=device)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # 标准 Attention
    start = time.perf_counter()
    output1, _ = standard_attention(Q, K, V)
    if device == "cuda":
        torch.cuda.synchronize()
    standard_time = time.perf_counter() - start

    if device == "cuda":
        standard_memory = torch.cuda.max_memory_allocated() / 1e6  # MB

        torch.cuda.reset_peak_memory_stats()

    # FlashAttention
    start = time.perf_counter()
    output2 = flash_attention(Q, K, V)
    if device == "cuda":
        torch.cuda.synchronize()
    flash_time = time.perf_counter() - start

    if device == "cuda":
        flash_memory = torch.cuda.max_memory_allocated() / 1e6

        return {
            "standard_time_ms": standard_time * 1000,
            "flash_time_ms": flash_time * 1000,
            "standard_memory_mb": standard_memory,
            "flash_memory_mb": flash_memory,
            "speedup": standard_time / flash_time if flash_time > 0 else 0,
            "memory_reduction": standard_memory / flash_memory if flash_memory > 0 else 0
        }
    else:
        return {
            "standard_time_ms": standard_time * 1000,
            "flash_time_ms": flash_time * 1000,
            "speedup": standard_time / flash_time if flash_time > 0 else 0
        }


def compare_outputs():
    """比较两种实现的输出是否接近"""
    batch_size, num_heads, seq_len, d_k = 2, 8, 128, 64

    Q = torch.randn(batch_size, num_heads, seq_len, d_k)
    K = torch.randn(batch_size, num_heads, seq_len, d_k)
    V = torch.randn(batch_size, num_heads, seq_len, d_k)

    output1, _ = standard_attention(Q, K, V)
    output2 = flash_attention(Q, K, V)

    # 计算差异
    diff = (output1 - output2).abs().max().item()
    relative_diff = diff / output1.abs().mean().item()

    print(f"最大绝对差异: {diff:.6f}")
    print(f"相对差异: {relative_diff:.6f}")
    print(f"输出接近: {diff < 1e-5}")


if __name__ == "__main__":
    # 比较输出
    print("=== 输出比较 ===")
    compare_outputs()

    # 基准测试
    print("\n=== 性能基准测试 ===")

    if torch.cuda.is_available():
        device = "cuda"
        print(f"使用设备: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = "cpu"
        print("使用设备: CPU")

    # 不同序列长度的测试
    seq_lens = [128, 256, 512, 1024, 2048]
    d_k = 64
    num_heads = 8
    batch_size = 4

    print(f"\n配置: batch={batch_size}, heads={num_heads}, d_k={d_k}")
    print("-" * 70)
    print(f"{'Seq Len':<10} {'Standard':<15} {'Flash':<15} {'Speedup':<10}")
    print("-" * 70)

    for seq_len in seq_lens:
        if device == "cuda":
            stats = benchmark_attention(batch_size, num_heads, seq_len, d_k, device)
            print(f"{seq_len:<10} {stats['standard_time_ms']:.2f} ms{'':<5} {stats['flash_time_ms']:.2f} ms{'':<5} {stats['speedup']:.2f}x")
        else:
            stats = benchmark_attention(batch_size, num_heads, seq_len, d_k, device)
            print(f"{seq_len:<10} {stats['standard_time_ms']:.2f} ms{'':<5} {stats['flash_time_ms']:.2f} ms{'':<5} {stats['speedup']:.2f}x")

    if device == "cuda":
        print("\n=== 显存对比 ===")
        for seq_len in [512, 1024, 2048]:
            stats = benchmark_attention(batch_size, num_heads, seq_len, d_k, device)
            print(f"Seq {seq_len}: Standard {stats['standard_memory_mb']:.0f} MB -> Flash {stats['flash_memory_mb']:.0f} MB ({stats['memory_reduction']:.1f}x reduction)")