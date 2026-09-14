import argparse
import os
import time
from llmbpe.basic import BasicTokenizer
from llmbpe.regex import RegexTokenizer


def train_tokenizer(input_path: str, vocab_size: int, model_type: str, output_prefix: str, verbose: bool = True):
    """
    Trains a Byte Pair Encoding (BPE) tokenizer on a text file corpus.
    
    Args:
        input_path: Path to the input text file.
        vocab_size: Target vocabulary size (must be > 256).
        model_type: 'basic' for BasicTokenizer, 'regex' for RegexTokenizer.
        output_prefix: Prefix path for saved .model and .vocab files.
        verbose: Whether to print merge details during training.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if vocab_size <= 256:
        raise ValueError(f"vocab_size must be greater than 256 (base UTF-8 byte count), got {vocab_size}")

    print(f"Loading corpus from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    raw_bytes = text.encode("utf-8")
    raw_len = len(raw_bytes)
    char_len = len(text)
    print(f"Corpus stats: {char_len:,} characters | {raw_len:,} UTF-8 bytes")

    # Select tokenizer implementation
    if model_type.lower() == "basic":
        tokenizer = BasicTokenizer()
    elif model_type.lower() == "regex":
        tokenizer = RegexTokenizer()
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'basic' or 'regex'.")

    print(f"\n--- Training {model_type.upper()} Tokenizer (Target Vocab Size: {vocab_size}) ---")
    start_time = time.time()
    tokenizer.train(text, vocab_size=vocab_size, verbose=verbose)
    elapsed = time.time() - start_time

    # Encode trained corpus to evaluate compression
    print("\n--- Evaluating Compression Ratio ---")
    encoded_ids = tokenizer.encode(text)
    encoded_len = len(encoded_ids)
    compression_ratio = raw_len / encoded_len if encoded_len > 0 else 0.0

    print(f"Training Completed in {elapsed:.2f} seconds")
    print(f"Original Byte Length : {raw_len:,} bytes")
    print(f"Encoded Token Count  : {encoded_len:,} tokens")
    print(f"Compression Ratio    : {compression_ratio:.2f}x (Bytes / Tokens)")

    # Ensure output directory exists
    out_dir = os.path.dirname(output_prefix)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Save model and vocabulary files
    tokenizer.save(output_prefix)
    print(f"\nSaved model artifacts:")
    print(f"  - Model File : {output_prefix}.model")
    print(f"  - Vocab File : {output_prefix}.vocab")


def main():
    parser = argparse.ArgumentParser(
        description="Train a custom LLM Byte Pair Encoding (BPE) Tokenizer on a text corpus."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="tests/elon.txt",
        help="Path to training text file (default: train_corpus.txt)"
    )
    parser.add_argument(
        "--vocab_size",
        "-v",
        type=int,
        default=300,
        help="Target vocabulary size including base bytes (default: 300)"
    )
    parser.add_argument(
        "--model_type",
        "-m",
        type=str,
        choices=["basic", "regex"],
        default="regex",
        help="Tokenizer architecture: 'basic' or 'regex' (default: regex)"
    )
    parser.add_argument(
        "--output_prefix",
        "-o",
        type=str,
        default="models/tokenizer",
        help="Output prefix path for .model and .vocab files (default: models/tokenizer)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress step-by-step merge output"
    )

    args = parser.parse_args()
    train_tokenizer(
        input_path=args.input,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        output_prefix=args.output_prefix,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
