"""
预训练脚本
用于从头训练小型 GPT 模型
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional
import math


class TextDataset(Dataset):
    """简单的文本数据集"""

    def __init__(self, tokens: List[int], seq_len: int = 128):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y


def process_batch(batch, device):
    """统一批处理：数据迁移到设备，构造输入和目标"""
    input_ids, targets = batch
    input_ids = input_ids.to(device)
    targets = targets.to(device)
    # TextDataset已返回正确的输入-目标对，直接使用
    return input_ids, targets


class CasualMaskDataLoader(DataLoader):
    """带 casual mask 的 DataLoader"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """创建 causal mask (True=保留，False=屏蔽)"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return ~mask  # 改为 True=保留


def evaluate_model(model, dataloader, device):
    """评估模型困惑度"""
    model.eval()
    total_loss = 0
    num_batches = 0

    loss_fct = nn.CrossEntropyLoss(ignore_index=0)

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = process_batch(batch, device)

            # 前向传播
            logits = model(inputs)

            # 计算损失
            loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

            total_loss += loss.item()
            num_batches += 1

    return math.exp(total_loss / num_batches)


def train(
    model,
    train_dataloader,
    eval_dataloader,
    device,
    epochs=10,
    lr=1e-4,
    max_grad_norm=1.0,
    warmup_steps=100,
    log_interval=10,
    eval_interval=100,
    save_path="checkpoint.pt"
):
    """训练循环"""

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
    )

    loss_fct = nn.CrossEntropyLoss(ignore_index=0)

    model.train()
    global_step = 0

    for epoch in range(epochs):
        for batch_idx, (input_ids, targets) in enumerate(train_dataloader):
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            # 输入为前n-1个token，目标为后n-1个token
            inputs = input_ids[:, :-1]

            # 前向传播
            logits = model(inputs)

            # 计算损失
            loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()

            global_step += 1

            # 日志
            if global_step % log_interval == 0:
                print(f"Epoch {epoch+1}, Step {global_step}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

            # 评估
            if global_step % eval_interval == 0:
                eval_ppl = evaluate_model(model, eval_dataloader, device)
                print(f"Eval perplexity: {eval_ppl:.4f}")
                model.train()

            # 保存
            if global_step % (eval_interval * 10) == 0:
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, save_path)
                print(f"Checkpoint saved to {save_path}")

    return model


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from code.model.gpt import create_gpt_config, GPTModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建配置
    config = create_gpt_config("gpt2-small")
    config["vocab_size"] = 10000  # 简化的词表

    # 创建模型
    model = GPTModel(**config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 创建随机数据
    random_tokens = list(torch.randint(0, 10000, (10000,)).numpy())

    # 数据集
    train_ds = TextDataset(random_tokens[:8000], seq_len=64)
    eval_ds = TextDataset(random_tokens[8000:], seq_len=64)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=8)

    # 训练（只训练几步用于测试）
    print("\n=== Training Test ===")
    train(model, train_loader, eval_loader, device, epochs=1, max_grad_norm=1.0)

    print("\n训练完成！")