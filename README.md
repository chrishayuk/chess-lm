# Chess-LM

A Python library for tokenizing chess games for language model training. This library provides efficient encoding of chess positions and moves into token sequences suitable for training transformer models on chess games.

## Features

- **UCI Move Tokenization**: Converts chess moves to unique integer IDs using UCI notation
- **State Encoding**: Encodes chess board states including castling rights, en passant, and side to move
- **Legal Move Masking**: Generate masks for legal moves in any position
- **PGN Processing**: Convert PGN files to tokenized sequences for training
- **Efficient Vocabulary**: ~4,096 move tokens + 320 state tokens for complete game representation

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
game = {
    "states": [
        {"stm_white": True, "castles": "KQkq", "ep_file": None},
        {"stm_white": False, "castles": "KQkq", "ep_file": "e"}
    ],
    "moves": ["e2e4", "e7e5"]
}
tokens = encode_game(game)  # Returns [state_token, move_token, state_token, move_token]
```

### Processing PGN Files

Convert PGN files to tokenized format for training:

```bash
python scripts/pgn_to_ids.py --pgn games.pgn --out games.jsonl
```

This creates a JSONL file with one game per line, each containing states and moves.

## Project Structure

```
chess-lm/
├── src/chess_lm/
│   └── tokenizer/        # Tokenizer module
│       ├── __init__.py   # Module exports
│       ├── moves.py      # UCI move catalog and ID mapping
│       ├── state_tokens.py  # Chess state encoding  
│       ├── encoding.py   # Game tokenization functions
│       ├── masks.py      # Legal move masking for training
│       └── cli.py        # CLI interface
├── scripts/
│   ├── pgn_to_ids.py     # PGN to token conversion
│   └── check_moves_and_tokenizer.py  # Validation utilities
└── tests/
    ├── test_moves.py      # Move encoding tests
    ├── test_tokenizer.py  # Tokenizer tests
    └── test_masks.py      # Legal move mask tests
```

## Core Components

### Move Vocabulary

The library uses a comprehensive catalog of all possible chess moves in UCI notation:
- All piece moves (including castling as king moves)
- All pawn promotions (queen, rook, bishop, knight)
- Total vocabulary: ~4,096 unique moves

### State Tokens

Board states are encoded with:
- Side to move (white/black)
- Castling rights (KQkq combinations)
- En passant file (if applicable)
- Total: 320 unique state tokens

### Token Sequence Format

Games are encoded as alternating state and move tokens:
```
[STATE_0, MOVE_0, STATE_1, MOVE_1, ..., STATE_N, MOVE_N]
```

This format preserves complete game information and enables autoregressive modeling.

## Testing

Run the test suite with pytest:

```bash
# Using uv
uv run pytest tests/ -v

# Using pytest directly
pytest tests/ -v
```

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

[Add your license here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.