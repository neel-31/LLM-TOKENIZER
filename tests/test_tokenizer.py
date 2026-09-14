import os
import tempfile
import unittest
from llmbpe.basic import BasicTokenizer
from llmbpe.regex import RegexTokenizer


def get_elon_full_text():
    """Helper function to load the entire elon.txt file content.
    Looks for data/elon.txt or elon.txt relative to the test script directory.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(base_dir, "..", "data", "elon.txt"),
        os.path.join(base_dir, "..", "elon.txt"),
        os.path.join(base_dir, "elon.txt"),
        "data/elon.txt",
        "elon.txt"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                if content.strip():
                    return content
                    
    raise FileNotFoundError(
        "Could not locate elon.txt dataset file. Please ensure elon.txt or data/elon.txt exists."
    )


class TestTokenizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load the complete elon.txt dataset once for all test cases."""
        cls.full_text = get_elon_full_text()

    def test_basic_tokenizer_full_file_roundtrip(self):
        """Test BasicTokenizer training, encoding, and exact round-trip decoding on the entire elon.txt file."""
        tokenizer = BasicTokenizer()
        target_vocab_size = 300  # 256 base byte tokens + 44 learned BPE merges

        # Train on the entire full document
        tokenizer.train(self.full_text, vocab_size=target_vocab_size, verbose=False)

        # Encode full text into token IDs
        encoded_ids = tokenizer.encode(self.full_text)

        # Decode token IDs back to string
        decoded_text = tokenizer.decode(encoded_ids)

        # Verify 100% exact round-trip identity across the entire file
        self.assertEqual(decoded_text, self.full_text)

        # Verify compression ratio (original byte count > token count)
        raw_byte_count = len(self.full_text.encode("utf-8"))
        token_count = len(encoded_ids)
        self.assertLess(token_count, raw_byte_count)
        
        compression_ratio = raw_byte_count / token_count
        self.assertGreater(compression_ratio, 1.1)

    def test_regex_tokenizer_full_file_roundtrip(self):
        """Test RegexTokenizer with GPT-4 regex pre-splitting on the entire elon.txt file."""
        tokenizer = RegexTokenizer()
        target_vocab_size = 300

        # Train on full document with regex chunk pre-splitting
        tokenizer.train(self.full_text, vocab_size=target_vocab_size, verbose=False)

        # Encode full document
        encoded_ids = tokenizer.encode(self.full_text)

        # Decode token IDs back to string
        decoded_text = tokenizer.decode(encoded_ids)

        # Verify 100% exact round-trip identity across the entire file
        self.assertEqual(decoded_text, self.full_text)

        # Verify compression
        raw_byte_count = len(self.full_text.encode("utf-8"))
        token_count = len(encoded_ids)
        self.assertLess(token_count, raw_byte_count)

    def test_regex_tokenizer_special_tokens(self):
        """Test special control token registration, encoding, and decoding using full file text."""
        tokenizer = RegexTokenizer()
        tokenizer.train(self.full_text, vocab_size=280, verbose=False)

        special_tokens = {
            "<|endoftext|>": 100257,
            "<|endofprompt|>": 100258
        }
        tokenizer.register_special_tokens(special_tokens)

        # Combine text from full document with special control tokens
        test_doc = f"<|endofprompt|>\n{self.full_text}\n<|endoftext|>"

        ids = tokenizer.encode(test_doc, allowed_special={"<|endoftext|>", "<|endofprompt|>"})

        # Assert special token IDs are present in the encoded sequence
        self.assertIn(100258, ids)
        self.assertIn(100257, ids)

        # Verify round-trip decoding restores original document including special tokens
        decoded = tokenizer.decode(ids)
        self.assertEqual(decoded, test_doc)

    def test_invalid_utf8_decoding_fallback(self):
        """Test decoding invalid UTF-8 byte sequences using errors='replace' fallback."""
        tokenizer = BasicTokenizer()

        # 128 (0x80) is an invalid standalone UTF-8 start byte
        invalid_byte_ids = [128]
        decoded = tokenizer.decode(invalid_byte_ids)

        # Replacement character \ufffd should be produced instead of throwing UnicodeDecodeError
        self.assertIn("\ufffd", decoded)

    def test_full_dataset_model_serialization(self):
        """Test saving vocabulary model trained on full dataset and loading it back."""
        tokenizer = BasicTokenizer()
        tokenizer.train(self.full_text, vocab_size=285, verbose=False)

        with tempfile.TemporaryDirectory() as tmp_dir:
            prefix = os.path.join(tmp_dir, "elon_full_bpe")
            tokenizer.save(prefix)

            self.assertTrue(os.path.exists(f"{prefix}.model"))
            self.assertTrue(os.path.exists(f"{prefix}.vocab"))

            # Load into fresh tokenizer instance
            loaded_tokenizer = BasicTokenizer()
            loaded_tokenizer.load(f"{prefix}.model")

            # Verify identical encoding behavior on full document
            self.assertEqual(tokenizer.encode(self.full_text), loaded_tokenizer.encode(self.full_text))


if __name__ == "__main__":
    unittest.main()
