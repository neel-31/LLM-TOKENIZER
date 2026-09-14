### LLM BPE Tokenizer
##### Custom Byte Pair Encoding Tokenizer for Large Language Models

An implementation of a Byte Pair Encoding (BPE) tokenizer built from scratch in Python. Designed for Large Language Model processing, this repository covers the full pipeline from raw UTF-8 byte stream conversion to iterative pair merging, regex pre-splitting, special token handling, and round-trip decoding.

#### Tech Stack
##### Core Library
    * Python 3.12
    * regex
##### Testing and Verification
    * pytest
    * tiktoken

#### System Architecture
```
                    Input Text Corpus
                           |
                           v
              UTF-8 Byte Stream Converter
                           |
                           v
              Regex Pre-Splitting Engine
                           |
                           v
              BPE Pair Statistics Engine
                           |
                           v
                 Iterative Merge Loop
                        /        \
                       /          \
                      v            v
             New Token IDs    Vocabulary & Merge Rules
                           \      /
                            v    v
                    Encoded Token IDs
                           |
                           v
             UTF-8 Byte Assembly Decoder
                           |
                           v
                    Restored Text
```

#### System Components
##### 1. Base Tokenizer (`minbpe/base.py`)
    * Provides shared vocabulary structures and merge rule dictionaries.
    * Implements pair frequency counting algorithms.
    * Handles sequence pair replacement logic.
##### 2. Basic BPE Tokenizer (`minbpe/basic.py`)
    * Converts raw text to UTF-8 byte sequences.
    * Iteratively merges frequent adjacent token pairs until reaching target vocabulary size.
    * Decodes integer token sequences back into strings using error-replacement fallback for invalid byte sequences.
##### 3. Regex BPE Tokenizer (`minbpe/regex.py`)
    * Applies regex split patterns prior to BPE merging.
    * Prevents merges across distinct character categories, numbers, and punctuation.
    * Handles explicit special control tokens such as end-of-text markers.
##### 4. Training Engine (`train.py`)
    * Ingests text files and runs vocabulary training workflows.
    * Reports compression ratio metrics comparing original byte counts to token counts.

#### Integration Pipeline
```
Raw Text Input
    |
    v
UTF-8 Byte Stream
    |
    v
Regex Pattern Chunking
    |
    v
BPE Merge Rule Application
    |
    v
Token ID Sequence
    |
    v
Token ID Lookup & Byte Concatenation
    |
    v
UTF-8 Safe String Decoding
    |
    v
Round-Trip Restoration
```

#### Project Structure
```
llm-bpe-tokenizer/
│
├── minbpe/
│   ├── __init__.py
│   ├── base.py
│   ├── basic.py
│   └── regex.py
│
├── tests/
│   └── test_tokenizer.py
│
├── train.py
├── requirements.txt
├── .gitignore
└── README.md
```

#### Running the Project
##### 1. Install Dependencies
From the project root:
```bash
pip install -r requirements.txt
```

##### 2. Train Tokenizer
```bash
python train.py --input data/sample.txt --vocab_size 300
```

##### 3. Run Verification Tests
```bash
pytest tests/
```

#### API Overview
##### BasicTokenizer
```python
tokenizer = BasicTokenizer()
tokenizer.train(text, vocab_size=276)
ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)
```

##### RegexTokenizer
```python
tokenizer = RegexTokenizer()
tokenizer.train(text, vocab_size=300)
ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
decoded = tokenizer.decode(ids)
```

#### Current Status
**Complete**
Verified:
```
UTF-8 Byte Encoding
Pair Frequency Statistics
Iterative BPE Merging
Regex Pre-Splitting
Special Token Interception
Round-Trip Text Restoration
tiktoken Benchmark Comparison
```

#### Security
Do not commit large training dataset binaries, raw model dumps, or API keys to GitHub.
Ensure `.gitignore` excludes temporary files and local virtual environments:
```
__pycache__/
*.pyc
.venv/
venv/
data/raw/
```

#### License
This project is intended for educational and development purposes.
