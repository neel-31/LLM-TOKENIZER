"""
Base Tokenizer class for Byte Pair Encoding (BPE).
Contains shared helper functions for pair statistics and merging.
"""

import unicodedata

def get_stats(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs.
    Optionally updates an existing counts dictionary.
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    """
    Given a list of integers, replace all consecutive occurrences of pair with idx.
    """
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def replace_control_characters(s: str) -> str:
    """
    Replaces non-printable control characters with unicode escape sequences
    so vocabulary files stay clean and readable.
    """
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] == "C":
            chars.append(f"\\u{ord(ch):04x}")
        else:
            chars.append(ch)
    return "".join(chars)

def render_token(t: bytes) -> str:
    """
    Pretty prints a token byte sequence as a readable string.
    """
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

class Tokenizer:
    """Base class for all Tokenizers."""

    def __init__(self):
        # default vocabulary is 256 raw bytes (0..255)
        self.merges = {}  # (int, int) -> int
        self.pattern = "" # regex pattern string
        self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab() # int -> bytes

    def train(self, text, vocab_size, verbose=False):
        """Train a vocabulary of size vocab_size from text."""
        raise NotImplementedError

    def encode(self, text):
        """Encode text string into a list of token IDs."""
        raise NotImplementedError

    def decode(self, ids):
        """Decode a list of token IDs back into a text string."""
        raise NotImplementedError

    def _build_vocab(self):
        """
        Reconstruct vocabulary mapping (int -> bytes) from raw bytes and merges.
        """
        vocab = {i: bytes([i]) for i in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix):
        """
        Saves two files: file_prefix.model (for loading) and file_prefix.vocab (for human inspection).
        """
        # Save model file
        model_file = file_prefix + ".model"
        with open(model_file, 'w', encoding='utf-8') as f:
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            for (p0, p1), idx in self.merges.items():
                f.write(f"{p0} {p1}\n")

        # Save vocab file for human visual inspection
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token_bytes in self.vocab.items():
                s = render_token(token_bytes)
                if idx in inverted_merges:
                    p0, p1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[p0])
                    s1 = render_token(self.vocab[p1])
                    f.write(f"[{s}] {idx} <- ({p0}, {p1}) [{s0}] [{s1}]\n")
                else:
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """Loads model file created with save()."""
        assert model_file.endswith(".model")
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding='utf-8') as f:
            version = f.readline().strip()
            assert version == "minbpe v1"
            self.pattern = f.readline().strip()
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            for line in f:
                p0, p1 = map(int, line.split())
                merges[(p0, p1)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()
