"""
Basic Byte Pair Encoding (BPE) Tokenizer implementation.
Converts raw UTF-8 byte streams and iteratively merges frequent adjacent pairs.
"""

try:
    from .base import Tokenizer, get_stats, merge
except ImportError:
    from base import Tokenizer, get_stats, merge

class BasicTokenizer(Tokenizer):
    """
    Minimal BPE Tokenizer without regex pre-splitting or special tokens.
    Operates directly on UTF-8 bytes.
    """

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        """
        Train BPE vocabulary on input text string up to target vocab_size.
        """
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # Convert text string to raw UTF-8 byte sequence
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        merges = {} # (p0, p1) -> idx
        vocab = {i: bytes([i]) for i in range(256)} # idx -> bytes

        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break
            
            # Find pair with highest frequency
            pair = max(stats, key=stats.get)
            if stats[pair] < 1:
                break

            idx = 256 + i
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        self.merges = merges
        self.vocab = vocab

    def encode(self, text):
        """
        Encode text string into token IDs using learned merge rules.
        """
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        while len(ids) >= 2:
            stats = get_stats(ids)
            # Find the pair in ids that was merged earliest during training
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break # No more mergeable pairs
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)

        return ids

    def decode(self, ids):
        """
        Decode list of token IDs back into text string.
        Fallback to errors='replace' for invalid UTF-8 byte sequences.
        """
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text
