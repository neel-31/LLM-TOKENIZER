import regex as re
try: 
    from .base import Tokenizer, get_stats, merge 
except ImportError: 
    from base import Tokenizer, get_stats, merge

# GPT-4 split pattern
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# GPT-2 split pattern
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):
    """
    Regex-based Byte Pair Encoding (BPE) Tokenizer.
    Enforces regex pre-splitting (GPT-2 / GPT-4 style) to prevent merges
    across letters, numbers, punctuation, or white space boundaries.
    Also supports special control tokens (e.g. <|endoftext|>).
    """

    def __init__(self, pattern=None):
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}        # str -> int, e.g. {"<|endoftext|>": 100257}
        self.inverse_special_tokens = {} # int -> str

    def register_special_tokens(self, special_tokens: dict):
        """
        Register special control tokens mapping string representation to token ID.
        Example: {"<|endoftext|>": 100257}
        """
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def _encode_chunk(self, text_bytes: list) -> list:
        """
        Encodes a single chunk of byte tokens using learned merges.
        """
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            # Find the pair with the lowest merge index (earliest merge)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # No more mergeable pairs
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        """
        Trains the tokenizer on input text using regex pre-chunking.
        Merges are computed across chunks independently without merging across chunk boundaries.
        """
        assert vocab_size >= 256, "vocab_size must be at least 256"
        num_merges = vocab_size - 256

        # Split text into chunks based on regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)

        # Convert each chunk to a list of UTF-8 byte integers
        ids_chunks = [list(chunk.encode("utf-8")) for chunk in text_chunks]

        # Iteratively merge the most frequent pair across all chunks
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            # Aggregate pair statistics across all chunks
            stats = {}
            for chunk in ids_chunks:
                for pair, count in get_stats(chunk).items():
                    stats[pair] = stats.get(pair, 0) + count

            if not stats:
                break

            # Find the most frequent pair
            top_pair = max(stats, key=stats.get)

            # Mint a new token ID
            idx = 256 + i
            # Merge the top pair in each chunk independently
            ids_chunks = [merge(chunk, top_pair, idx) for chunk in ids_chunks]

            # Record the merge rule
            merges[top_pair] = idx
            vocab[idx] = vocab[top_pair[0]] + vocab[top_pair[1]]

            if verbose:
                print(f"merge {i+1}/{num_merges}: {top_pair} -> {idx} ({vocab[idx]!r}) had {stats[top_pair]} occurrences")

        self.merges = merges
        self.vocab = vocab

    def encode_ordinary(self, text: str) -> list:
        """
        Encodes ordinary text by splitting with regex and applying learned merges to each chunk.
        Ignores special tokens.
        """
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text: str, allowed_special: str | set = "none") -> list:
        """
        Encodes text, handling special tokens appropriately.
        allowed_special can be:
          - "all": allow all registered special tokens
          - "none": disallow all special tokens (raise ValueError if any are present)
          - set of str: explicit set of special tokens allowed in text
        """
        # Determine allowed special tokens set
        if allowed_special == "all":
            special_allowed = set(self.special_tokens.keys())
        elif allowed_special == "none":
            special_allowed = set()
        elif isinstance(allowed_special, set):
            special_allowed = allowed_special
        else:
            raise ValueError(f"Invalid allowed_special option: {allowed_special}")

        # Check for disallowed special tokens
        disallowed = set(self.special_tokens.keys()) - special_allowed
        if disallowed:
            # Build regex pattern matching any disallowed special token
            disallowed_pattern = "(" + "|".join(re.escape(tok) for tok in disallowed) + ")"
            if re.search(disallowed_pattern, text):
                raise ValueError(f"Disallowed special token found in text. Allowed: {special_allowed}")

        if not special_allowed:
            return self.encode_ordinary(text)

        # Build regex pattern to split text by allowed special tokens
        special_pattern = "(" + "|".join(re.escape(tok) for tok in special_allowed) + ")"
        special_chunks = re.split(special_pattern, text)

        ids = []
        for part in special_chunks:
            if part in special_allowed:
                ids.append(self.special_tokens[part])
            elif part:
                ids.extend(self.encode_ordinary(part))

        return ids

    def decode(self, ids: list) -> str:
        """
        Decodes a list of token IDs back into a UTF-8 string.
        Handles both vocabulary byte tokens and special tokens.
        """
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"Invalid token ID: {idx}")

        full_bytes = b"".join(part_bytes)
        return full_bytes.decode("utf-8", errors="replace")
