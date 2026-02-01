# Chess-LM

A Python library for tokenizing chess games for language model training. This library provides efficient encoding of chess positions and moves into token sequences suitable for training transformer models on chess games.

## Features

- **UCI Move Tokenization**: Converts chess moves to unique integer IDs using UCI notation
- **State Encoding**: Encodes chess board states including castling rights, en passant, and side to move
- **Legal Move Masking**: Generate masks for legal moves in any position
- **PGN Processing**: Convert PGN files to tokenized sequences for training
- **Efficient Vocabulary**: 4,208 move tokens + 288 state tokens for complete game representation

## Installation

### Using uv (recommended)

```bash
# Install the package in development mode
uv pip install -e .

# Install with dev dependencies
uv pip install -e ".[dev]"
```

### Using pip

```bash
# Install the package in development mode
pip install -e .
```

## Quick Start

### Basic Usage

```python
from chess_lm.tokenizer import uci_to_id, id_to_uci, vocab_size, encode_game

# Convert moves to token IDs
move_id = uci_to_id("e2e4")
move_uci = id_to_uci(move_id)

# Encode a complete game
game = {"moves": ["e2e4", "e7e5"]}
tokens = encode_game(game)  # Returns [state_token, move_token, state_token, move_token]
```

### Processing PGN Files

Convert PGN files to tokenized format for training:

```bash
python scripts/pgn_to_ids.py --pgn games.pgn --out games.jsonl
```

This creates a JSONL file with one game per line, each containing states and moves.

### Using the Dataset

Load tokenized games for model training:

```python
from chess_lm.data import ChessSequenceDataset, CollateCausal
from torch.utils.data import DataLoader

# Create dataset with sliding windows
dataset = ChessSequenceDataset(
    path="games.jsonl",
    max_len=128,           # Maximum sequence length
    stride=64,             # Stride for sliding window
    start_on_state=True,   # Ensure sequences start with state tokens
    drop_trailing_state=True  # Drop incomplete state-move pairs
)

# Create dataloader with custom collator
collator = CollateCausal(
    pad_id=vocab_total_size() - 1,
    mask_mode="only_moves"  # Only train on move predictions
)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collator
)

# Training loop
for batch in dataloader:
    input_ids = batch["input_ids"]  # [batch_size, seq_len]
    labels = batch["labels"]        # [batch_size, seq_len]
    loss_mask = batch["loss_mask"]  # [batch_size, seq_len]
```

## Project Structure

```
chess-lm/
├── src/chess_lm/
│   ├── tokenizer/        # Tokenizer module
│   │   ├── __init__.py   # Module exports
│   │   ├── moves.py      # UCI move catalog and ID mapping
│   │   ├── state_tokens.py  # Chess state encoding  
│   │   ├── encoding.py   # Game tokenization functions
│   │   ├── masks.py      # Legal move masking for training
│   │   └── cli.py        # CLI interface
│   └── data/            # Data processing module
│       ├── __init__.py   # Module exports
│       ├── chess_sequence_dataset.py  # PyTorch dataset for chess sequences
│       └── collate_causal.py  # Custom collator for language model training
├── scripts/
│   ├── pgn_to_ids.py     # PGN to token conversion
│   └── check_moves_and_tokenizer.py  # Validation utilities
├── tests/
│   ├── tokenizer/        # Tokenizer tests
│   │   ├── test_moves.py      # Move encoding tests
│   │   ├── test_tokenizer.py  # Tokenizer tests
│   │   ├── test_state_tokens.py  # State token tests
│   │   ├── test_masks.py      # Legal move mask tests
│   │   └── test_tokenizer_cli.py  # CLI tests
│   └── data/            # Data module tests
│       ├── test_chess_sequence_dataset.py  # Dataset tests
│       └── test_collate_causal.py  # Collator tests
└── Makefile             # Development workflow automation
```

## Core Components

### Move Vocabulary

The library uses a comprehensive catalog of all possible chess moves in UCI notation:
- All piece moves (including castling as king moves)
- All pawn promotions (queen, rook, bishop, knight)
- Total vocabulary: 4,208 unique moves

### State Tokens

Board states are encoded with:
- Side to move (white/black)
- Castling rights (KQkq combinations)
- En passant file (if applicable)
- Total: 288 unique state tokens

### Token Sequence Format

Games are encoded as alternating state and move tokens:
```
[STATE_0, MOVE_0, STATE_1, MOVE_1, ..., STATE_N, MOVE_N]
```

This format preserves complete game information and enables autoregressive modeling.

### Data Processing

#### ChessSequenceDataset

PyTorch dataset for creating training sequences from tokenized chess games:
- Configurable sequence length and stride for sliding windows
- Support for padding short sequences
- Options to start sequences on state tokens and drop trailing states
- Lazy loading for memory-efficient processing of large datasets
- Built-in caching for faster data loading

#### CollateCausal

Custom PyTorch collator for batching chess sequences:
- Automatic padding to maximum sequence length in batch
- Support for multiple masking modes (all, only_moves, boundary)
- Special token handling for simulation boundaries
- Efficient tensor operations for model training

## Development

### Testing

Run the test suite with pytest:

```bash
# Using make (recommended)
make test          # Run all tests
make test-verbose  # Run tests with verbose output
make coverage      # Run tests with coverage report

# Using uv directly
uv run pytest tests/ -v

# Using pytest directly
pytest tests/ -v
```

### Code Quality

```bash
# Run all checks
make check  # Runs format, lint, typecheck, and tests

# Individual checks
make format     # Format code with black and isort
make lint       # Check code with ruff
make type-check # Type checking with mypy
```

### Test Coverage

The project maintains high test coverage:
- Overall: 91% coverage
- Tokenizer module: 90–100% coverage
- Data module: 89% coverage

## Use Cases

- **Chess Engine Training**: Train transformer models to play chess
- **Game Analysis**: Analyze patterns in large chess databases
- **Move Prediction**: Build models that predict next moves
- **Position Evaluation**: Learn position evaluation from game outcomes

## Dependencies

- `python-chess`: Chess board representation and move generation
- `torch`: Tensor operations for masking
- `numpy`: Numerical operations
- `tqdm`: Progress bars for batch processing

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.