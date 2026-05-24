"""
BPE Tokenizer 实现
基于 GPT-2 的原始实现

功能:
- 训练 BPE 词表
- 编码文本为 token ids
- 解码 token ids 为文本
"""

from collections import Counter, defaultdict
import re
from typing import List, Tuple, Dict


class BPETokenizer:
    """BPE Tokenizer 实现"""

    def __init__(self, vocab_size: int = 5000, dropout: int = None):
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.merges: List[Tuple[int, int]] = []  # 记录所有合并操作
        self.vocab: Dict[int, bytes] = {}  # 最终词表

    def get_stats(self, words: Counter) -> Counter:
        """统计所有字符对的频率"""
        pairs = Counter()
        for word, freq in words.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def merge_vocab(self, words: Counter, pair: Tuple[int, int]) -> Counter:
        """合并所有词中的指定字符对"""
        new_words = Counter()
        first, second = pair

        for word, freq in words.items():
            new_word = word.replace(f'{first} {second}', f'{first}{second}')
            new_words[new_word] = freq

        return new_words

    def train(self, text: str, verbose: bool = True):
        """
        训练 BPE 词表

        参数:
            text: 训练文本
            verbose: 是否打印训练过程
        """
        # 1. 初始化：将文本转为字节序列
        words = Counter()
        for line in text.split('\n'):
            if not line.strip():
                continue
            tokens = list(line.encode('utf-8'))
            word = ' '.join([str(t) for t in tokens])
            words[word] += 1

        if verbose:
            print(f"初始词汇量: {len(words)}")

        # 2. 迭代合并
        for i in range(self.vocab_size - 256):
            pairs = self.get_stats(words)
            if not pairs:
                break

            best = pairs.most_common(1)[0][0]
            words = self.merge_vocab(words, best)
            self.merges.append(best)

            if verbose and (i + 1) % 1000 == 0:
                print(f"已合并 {i + 1} 次, 当前词表大小: {256 + i + 1}")

        # 3. 构建最终词表
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]

    def encode(self, text: str) -> List[int]:
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

    def decode(self, ids: List[int]) -> str:
        """对 token ids 进行解码"""
        return b''.join([bytes([id_]) for id_ in ids]).decode('utf-8', errors='replace')

    def save(self, path: str):
        """保存词表和合并规则"""
        import json
        data = {
            'vocab': {str(k): v.hex() for k, v in self.vocab.items()},
            'merges': [[a, b] for a, b in self.merges]
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path: str):
        """加载词表和合并规则"""
        import json
        with open(path, 'r') as f:
            data = json.load(f)

        self.vocab = {int(k): bytes.fromhex(v) for k, v in data['vocab'].items()}
        self.merges = [tuple(m) for m in data['merges']]


def get_encoder(model_name: str = "gpt2"):
    """获取预训练的 GPT-2 encoder"""
    import tiktoken
    return tiktoken.get_encoding(model_name)


if __name__ == "__main__":
    # 测试代码
    text = """
    DeepSeek is amazing! The model understands context very well.
    I love using DeepSeek for coding tasks. It's truly revolutionary.
    """

    bpe = BPETokenizer(vocab_size=2000)
    bpe.train(text)

    # 测试编码
    sentence = "DeepSeek"
    ids = bpe.encode(sentence)
    print(f"'{sentence}' 编码为: {ids}")

    # 测试解码
    decoded = bpe.decode(ids)
    print(f"解码回: '{decoded}'")

    # 使用 tiktoken
    print("\n使用 tiktoken:")
    enc = get_encoder("gpt2")
    tokens = enc.encode("DeepSeek is fantastic!")
    print(f"GPT-2 编码: {tokens}")
    print(f"GPT-2 解码: {enc.decode(tokens)}")