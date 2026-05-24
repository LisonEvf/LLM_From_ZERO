"""
BPE Tokenizer 实现
基于标准BPE算法
"""

from collections import Counter
from typing import List, Tuple, Dict


class BPETokenizer:
    """BPE Tokenizer 实现"""

    def __init__(self, vocab_size: int = 5000, dropout: int = None):
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.merges: List[Tuple[int, int]] = []  # 合并规则列表
        self.vocab: Dict[int, bytes] = {}  # 最终词表

    def train(self, text: str, verbose: bool = True):
        """训练 BPE 词表"""
        # 初始化词表：256个字节
        self.vocab = {i: bytes([i]) for i in range(256)}

        # 将文本转为字节序列元组
        words = Counter()
        for line in text.split('\n'):
            if not line.strip():
                continue
            tokens = tuple(line.encode('utf-8'))
            words[tokens] += 1

        if verbose:
            print(f"初始词汇量: {len(words)}")

        # 迭代合并
        for i in range(self.vocab_size - 256):
            # 统计所有字符对的频率
            pair_counts = Counter()
            for word_tokens, freq in words.items():
                for j in range(len(word_tokens) - 1):
                    pair = (word_tokens[j], word_tokens[j + 1])
                    pair_counts[pair] += freq

            if not pair_counts:
                break

            # 贪心选择最高频的字符对
            best_pair = pair_counts.most_common(1)[0][0]
            self.merges.append(best_pair)

            # 合并所有词中的该字符对
            new_words = Counter()
            a, b = best_pair
            for word_tokens, freq in words.items():
                new_tokens = []
                j = 0
                while j < len(word_tokens):
                    if j < len(word_tokens) - 1 and word_tokens[j] == a and word_tokens[j + 1] == b:
                        j += 2
                    else:
                        new_tokens.append(word_tokens[j])
                        j += 1
                new_words[tuple(new_tokens)] = freq
            words = new_words

            if verbose and (i + 1) % 1000 == 0:
                print(f"已合并 {i + 1} 次, 当前词表大小: {256 + i + 1}")

        # 构建词表
        for i, (a, b) in enumerate(self.merges):
            # a和b是原始字节值(0-255)，直接用bytes([a]) + bytes([b])构建
            self.vocab[256 + i] = bytes([a]) + bytes([b])

    def encode(self, text: str) -> List[int]:
        """编码文本"""
        tokens = list(text.encode('utf-8'))

        # 持续查找可合并的字符对并合并
        changed = True
        while changed:
            changed = False
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1:
                    pair = (tokens[i], tokens[i + 1])
                    if pair in self.merges:
                        merge_idx = self.merges.index(pair)
                        new_tokens.append(256 + merge_idx)
                        i += 2
                        changed = True
                        continue
                new_tokens.append(tokens[i])
                i += 1
            tokens = new_tokens

        return tokens

    def decode(self, ids: List[int]) -> str:
        """解码 token ids"""
        result = []
        for id_ in ids:
            if id_ < 256:
                result.append(id_)
            else:
                idx = id_ - 256
                if 0 <= idx < len(self.merges):
                    a, b = self.merges[idx]
                    result.extend([a, b])
                else:
                    result.append(id_)
        return bytes(result).decode('utf-8', errors='replace')

    def save(self, path: str):
        """保存词表"""
        import json
        data = {
            'vocab': {str(k): v.hex() for k, v in self.vocab.items()},
            'merges': [[a, b] for a, b in self.merges]
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path: str):
        """加载词表"""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        self.vocab = {int(k): bytes.fromhex(v) for k, v in data['vocab'].items()}
        self.merges = [tuple(m) for m in data['merges']]


if __name__ == "__main__":
    text = "hello world hello world hello"
    bpe = BPETokenizer(vocab_size=300)
    bpe.train(text, verbose=False)

    ids = bpe.encode("hello")
    print(f"'hello' encoded: {ids}")

    decoded = bpe.decode(ids)
    print(f"decoded: '{decoded}'")
    print(f"correct: {decoded == 'hello'}")