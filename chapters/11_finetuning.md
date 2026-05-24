# 第十一章：微调的艺术 — SFT、RLHF与LoRA

## 故事开场：预训练之后

Bob完成了一个7B参数的预训练模型。但当他让模型回答"中国的首都是什么"时：

```
输入: "中国的首都是"
输出: "中国的首都是一个人口众多的大城市，拥有悠久的历史..."  ← 长篇大论！
```

"为什么模型无法简洁回答？"Bob困惑。

Alice解释："预训练的任务是**预测下一个token**，它学会了'续写'，但没学会'回答问题'。这就是微调的作用。"

## 11.1 微调概述

### 什么是微调？

**微调（Fine-tuning）**：在预训练模型基础上，用少量标注数据让模型适应特定任务。

```
预训练模型:
- 任务: 预测下一个token
- 数据: 大规模无标注文本（万亿tokens）
- 目标: 学习通用语言表示

微调模型:
- 任务: 问答、分类、生成等
- 数据: 少量标注数据（几千~几万样本）
- 目标: 适配特定任务
```

### 微调类型

| 类型 | 数据需求 | 计算成本 | 效果 |
|------|----------|----------|------|
| 全参数微调 | 高 | 高 | 好，但慢 |
| LoRA | 低 | 低 | 接近全参数 |
| QLoRA | 很低 | 很低 | 接近全参数 |
| Adapter | 低 | 中 | 中等 |

## 11.2 SFT: 监督微调

SFT（Supervised Fine-Tuning）是最直接的微调方式。

### 流程

```python
# code/training/sft.py
"""
SFT 监督微调
"""

import torch
from torch.utils.data import DataLoader

def sft_train(model, train_dataset, epochs=3, lr=1e-5):
    """
    监督微调流程

    参数:
        model: 预训练模型
        train_dataset: 训练数据 (prompt, response) 对
        epochs: 训练轮数
        lr: 学习率
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in DataLoader(train_dataset, batch_size=4):
            # 构造训练样本
            # 输入: prompt + response
            # 目标: 只预测response部分的loss
            inputs = batch['input_ids']
            labels = batch['labels']

            # 前向传播
            logits = model(inputs)

            # 计算loss（只对response部分）
            loss = compute_loss(logits, labels, pad_token_id=0)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_dataset):.4f}")

    return model
```

### 数据格式

```python
# 训练数据格式
train_data = [
    {
        "prompt": "中国的首都是哪里？",
        "response": "中国的首都是北京。"
    },
    {
        "prompt": "解释一下什么是机器学习",
        "response": "机器学习是人工智能的一个分支..."
    }
]

# 构造训练样本
def preprocess_example(example):
    """
    构造指令微调格式

    <|user|>
    中国的首都是哪里？
    <|assistant|>
    中国的首都是北京。
    """
    text = f"<|user|>\n{example['prompt']}\n<|assistant|>\n{example['response']}"
    return text
```

## 11.3 RLHF: 人类反馈强化学习

SFT的问题是：**它只学习"正确的回答"，不学习"为什么正确"**。

RLHF（Reinforcement Learning from Human Feedback）通过人类反馈来学习偏好。

### RLHF三步骤

```
┌────────────────────────────────────────────────────────────┐
│                    RLHF 流程                                │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  步骤1: 预训练 → SFT                                       │
│         基础模型 + 人工标注数据 → SFT模型                   │
│                                                            │
│  步骤2: 奖励模型 (Reward Model)                           │
│         人类对比数据 → 训练RM预测人类偏好                   │
│                                                            │
│  步骤3: PPO强化学习                                        │
│         SFT模型 + RM → 优化生成质量                        │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 步骤2：训练奖励模型

```python
# 训练Reward Model
def train_reward_model(sft_model, preference_data):
    """
    训练奖励模型

    preference_data: [(prompt, response_a, response_b, label)]
    label: 哪个更好 (0=a更好, 1=b更好)
    """
    reward_model = RewardModel(sft_model)

    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=1e-5)

    for epoch in range(3):
        for prompt, resp_a, resp_b, label in preference_data:
            # 分别获取两个response的奖励
            reward_a = reward_model(prompt, resp_a)
            reward_b = reward_model(prompt, resp_b)

            # 对比损失：更喜欢的response应该有更高奖励
            loss = -torch.log(torch.sigmoid(reward_b - reward_a)) * label

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return reward_model
```

### 步骤3：PPO强化学习

```python
# code/training/rlhf.py
"""
PPO强化学习
"""

def train_with_ppo(sft_model, reward_model, prompts):
    """
    PPO训练流程
    """
    ref_model = copy.deepcopy(sft_model)  # 参考模型（固定）
    optimizer = torch.optim.AdamW(sft_model.parameters(), lr=1e-6)

    for iteration in range(100):
        # 1. 用当前模型生成response
        responses = []
        for prompt in prompts:
            response = sft_model.generate(prompt, max_len=100)
            responses.append(response)

        # 2. 计算奖励
        rewards = []
        for prompt, response in zip(prompts, responses):
            reward = reward_model(prompt, response)
            rewards.append(reward)

        # 3. PPO更新
        for prompt, response, reward in zip(prompts, responses, rewards):
            # 计算log prob
            log_probs = sft_model.get_log_probs(prompt, response)

            # 参考模型的log prob（不能偏离太远）
            ref_log_probs = ref_model.get_log_probs(prompt, response)

            # PPO目标：最大化奖励 + KL惩罚
            # reward - beta * KL(ref || current)
            loss = -(reward - 0.1 * kl_divergence(log_probs, ref_log_probs))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return sft_model
```

## 11.4 LoRA: 低秩适配

全参数微调的问题：**参数太多，计算和存储成本高**。

LoRA（Low-Rank Adaptation）的核心思想：

> **与其更新所有参数，不如只更新一小部分低秩矩阵。**

### LoRA原理

```
预训练权重: W₀ (d × k)
更新: ΔW = BA (d × k), 其中 B是(d×r), A是(r×k), r << min(d,k)
新权重: W = W₀ + ΔW

训练时: 只训练 A 和 B
推理时: 合并到 W₀，零开销
```

```python
# code/training/lora.py
"""
LoRA 实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """LoRA适配的线性层"""
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 原始权重（冻结）
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features),
            requires_grad=False
        )
        if self.bias is not None:
            self.bias = nn.Parameter(
                torch.zeros(out_features),
                requires_grad=False
            )
        else:
            self.register_parameter('bias', None)

        # LoRA参数（A和B）
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x):
        # 原始输出
        output = F.linear(x, self.weight, self.bias)

        # LoRA更新
        # W = W₀ + (alpha/rank) * B @ A
        lora_update = (self.lora_B @ self.lora_A) * self.scaling
        output = output + x @ lora_update.t()

        return output


class LoRAModel(nn.Module):
    """带LoRA的模型"""
    def __init__(self, base_model, rank=4, target_modules=['q_proj', 'v_proj']):
        super().__init__()
        self.base_model = base_model
        self.rank = rank

        # 替换目标层
        for name, module in base_model.named_modules():
            if any(target in name for target in target_modules):
                if isinstance(module, nn.Linear):
                    # 创建LoRA版本
                    lora_layer = LoRALinear(
                        module.in_features,
                        module.out_features,
                        rank=rank
                    )
                    # 复制原权重
                    lora_layer.weight.data = module.weight.data.clone()
                    if module.bias is not None:
                        lora_layer.bias.data = module.bias.data.clone()

                    # 替换
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    parent = self._get_module(parent_name)
                    setattr(parent, child_name, lora_layer)

    def _get_module(self, name):
        return self.base_model

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)


# 使用示例
from transformers import AutoModelForCausalLM

# 加载预训练模型
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")

# 应用LoRA
lora_model = LoRAModel(model, rank=4, target_modules=['q_proj', 'v_proj'])

# 统计可训练参数
trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in lora_model.parameters())
print(f"可训练参数: {trainable_params / 1e6:.2f}M / {total_params / 1e9:.2f}B")
# 输出: 约 4M / 7B = 0.06% 的参数需要训练！
```

## 11.5 QLoRA: 量化LoRA

QLoRA = 量化 + LoRA，进一步降低内存。

```python
# 使用 bitsandbytes 进行QLoRA
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig

# 量化配置：4位量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    quantization_config=bnb_config
)

# 配置LoRA
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM'
)

# 应用LoRA
model = get_peft_model(model, lora_config)

# 训练（显存需求大幅降低）
# 7B模型只需要约 6GB VRAM (原本需要 14GB)
```

## 11.6 微调实践

### 使用 HuggingFace TRL

```python
# 使用SFTTrainer进行SFT
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")

# 配置训练
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    fp16=True,
    logging_steps=10,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=lambda data: tokenizer(data, return_tensors="pt")
)

trainer.train()
```

### 使用 RLHF

```python
# 使用TRL的PPO trainer
from trl import PPOConfig, PPOTrainer

ppo_config = PPOConfig(
    model_name="meta-llama/Llama-2-7b",
    learning_rate=1e-5,
    batch_size=8,
    mini_batch_size=2,
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    reward_model=reward_model,
)

# 训练
ppo_trainer.train()
```

## 11.7 微调数据集格式

### 通用格式

```python
# 标准对话格式
{
    "messages": [
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "回答"}
    ]
}

# Alpaca格式
{
    "instruction": "回答问题",
    "input": "上下文（可选）",
    "output": "回答"
}

# ShareGPT格式
{
    "conversations": [
        {"from": "human", "value": "问题"},
        {"from": "gpt", "value": "回答"}
    ]
}
```

## 11.8 本章小结

1. **SFT**：用人工标注数据直接微调，最简单直接
2. **RLHF**：通过人类反馈学习偏好，更符合人类价值观
3. **LoRA**：低秩分解，大幅降低微调参数量
4. **QLoRA**：量化+LoRA，可在消费级GPU上微调7B模型
5. **选择建议**：快速实验用LoRA，对质量要求高用RLHF

### 思考题

1. SFT和RLHF的根本区别是什么？为什么RLHF通常效果更好？
2. LoRA的rank参数如何选择？rank太高或太低会有什么影响？
3. 为什么QLoRA能够保持接近全参数微调的效果？

### 延伸阅读

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLORA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)
- [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory)