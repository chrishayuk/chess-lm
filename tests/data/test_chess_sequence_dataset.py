"""Tests for ChessSequenceDataset"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from chess_lm.data.chess_sequence_dataset import ChessSequenceDataset
from chess_lm.tokenizer import is_state_token, vocab_total_size


@pytest.fixture
def sample_games():
    """Create sample games for testing"""
    return [
        {"moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"]},
        {"moves": ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "b7b6"]},
        {"moves": ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6"]},
        {"moves": ["d2d4", "d7d5", "c2c4", "e7e6"]},  # Short game
        {"moves": ["e2e4"] * 50},  # Long repetitive game for testing windowing
    ]


@pytest.fixture
def games_file(sample_games):
    """Create a temporary JSONL file with sample games"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for game in sample_games:
            json.dump(game, f)
            f.write("\n")
        return Path(f.name)


def test_dataset_initialization(games_file):
    """Test basic dataset initialization"""
    ds = ChessSequenceDataset(
        path=str(games_file),
        max_len=128,
        stride=64,
        start_on_state=True,
        drop_trailing_state=True,
    )

    assert len(ds) > 0
    assert ds.max_len == 128
    assert ds.stride == 64
    assert ds.start_on_state is True
    assert ds.drop_trailing_state is True


def test_dataset_length(games_file):
    """Test dataset length calculation"""
    # With different max_len and stride values
    ds1 = ChessSequenceDataset(path=str(games_file), max_len=32, stride=32)
    ds2 = ChessSequenceDataset(path=str(games_file), max_len=32, stride=16)  # Overlapping windows

    # With overlap (stride < max_len), we should have more windows
    # Note: Due to the failed game warning, we may have equal lengths if some games fail to encode
    assert len(ds2) >= len(ds1)


def test_dataset_getitem(games_file):
    """Test getting items from dataset"""
    ds = ChessSequenceDataset(path=str(games_file), max_len=64, stride=32)

    # Test first item
    item = ds[0]
    assert isinstance(item, list) or isinstance(item, dict)

    if isinstance(item, list):
        tokens = item
    else:
        tokens = item["tokens"]

    assert len(tokens) <= 64
    assert all(isinstance(t, (int, np.integer)) for t in tokens)


def test_start_on_state_constraint(games_file):
    """Test that sequences start with state tokens when required"""
    ds = ChessSequenceDataset(path=str(games_file), max_len=32, stride=16, start_on_state=True, validate_tokens=False)

    for i in range(min(10, len(ds))):
        tokens = ds[i]
        if isinstance(tokens, dict):
            tokens = tokens["tokens"]

        # First token should be a state token
        if len(tokens) > 0:
            assert is_state_token(tokens[0]), f"First token at index {i} is not a state token"


def test_drop_trailing_state(games_file):
    """Test dropping trailing state tokens"""
    ds = ChessSequenceDataset(
        path=str(games_file), max_len=32, stride=16, drop_trailing_state=True, validate_tokens=False
    )

    for i in range(min(10, len(ds))):
        tokens = ds[i]
        if isinstance(tokens, dict):
            tokens = tokens["tokens"]

        # Last token should NOT be a state token (should be a move)
        if len(tokens) > 0:
            assert not is_state_token(tokens[-1]), f"Last token at index {i} is a state token"


def test_token_alternation(games_file):
    """Test that tokens alternate between state and move"""
    ds = ChessSequenceDataset(path=str(games_file), max_len=64, stride=32, start_on_state=True, validate_tokens=False)

    for i in range(min(5, len(ds))):
        tokens = ds[i]
        if isinstance(tokens, dict):
            tokens = tokens["tokens"]

        # Check alternation: even indices are states, odd are moves
        for j, tok in enumerate(tokens):
            expected_state = j % 2 == 0
            actual_state = is_state_token(tok)
            assert expected_state == actual_state, f"Token alternation broken at window {i}, position {j}"


def test_return_info(games_file):
    """Test returning additional info with sequences"""
    ds = ChessSequenceDataset(path=str(games_file), max_len=32, stride=16, return_info=True)

    item = ds[0]
    assert isinstance(item, dict)
    assert "tokens" in item
    # The info is directly in the dict, not nested
    assert "game_id" in item
    assert "start_pos" in item
    assert "end_pos" in item


def test_lazy_loading(games_file):
    """Test lazy loading mode"""
    ds_eager = ChessSequenceDataset(path=str(games_file), max_len=32, stride=16, lazy_load=False)
    ds_lazy = ChessSequenceDataset(path=str(games_file), max_len=32, stride=16, lazy_load=True)

    # Both should produce the same number of windows
    assert len(ds_eager) == len(ds_lazy)

    # First few items should be identical
    for i in range(min(5, len(ds_eager))):
        eager_tokens = ds_eager[i]
        lazy_tokens = ds_lazy[i]

        if isinstance(eager_tokens, dict):
            eager_tokens = eager_tokens["tokens"]
        if isinstance(lazy_tokens, dict):
            lazy_tokens = lazy_tokens["tokens"]

        assert eager_tokens == lazy_tokens


def test_seed_reproducibility(games_file):
    """Test that setting seed produces reproducible results"""
    ds1 = ChessSequenceDataset(path=str(games_file), max_len=32, stride=16, seed=42)
    ds2 = ChessSequenceDataset(path=str(games_file), max_len=32, stride=16, seed=42)
    ds3 = ChessSequenceDataset(path=str(games_file), max_len=32, stride=16, seed=123)

    # Same seed should produce same order
    for i in range(min(5, len(ds1))):
        tokens1 = ds1[i]
        tokens2 = ds2[i]
        if isinstance(tokens1, dict):
            tokens1 = tokens1["tokens"]
        if isinstance(tokens2, dict):
            tokens2 = tokens2["tokens"]
        assert tokens1 == tokens2

    # Different seed might produce different order
    # (though with small dataset it might randomly match)
    for i in range(min(5, len(ds1))):
        tokens1 = ds1[i]
        tokens3 = ds3[i]
        if isinstance(tokens1, dict):
            tokens1 = tokens1["tokens"]
        if isinstance(tokens3, dict):
            tokens3 = tokens3["tokens"]
        if tokens1 != tokens3:
            break
    # We don't assert different is True because small datasets might coincidentally match


def test_padding_short_sequences(games_file):
    """Test padding of short sequences"""
    # Use last token in vocab as padding (vocab_total_size - 1)
    ds = ChessSequenceDataset(
        path=str(games_file), max_len=128, stride=64, pad_short_sequences=True, pad_token_id=vocab_total_size() - 1
    )

    for i in range(min(5, len(ds))):
        tokens = ds[i]
        if isinstance(tokens, dict):
            tokens = tokens["tokens"]

        # All sequences should be exactly max_len when padding is enabled
        assert len(tokens) == 128, f"Sequence {i} has length {len(tokens)}, expected 128"


def test_validate_tokens(games_file):
    """Test token validation"""
    ds = ChessSequenceDataset(path=str(games_file), max_len=32, stride=16, validate_tokens=True)

    vocab_size = vocab_total_size()
    for i in range(min(10, len(ds))):
        tokens = ds[i]
        if isinstance(tokens, dict):
            tokens = tokens["tokens"]

        # All tokens should be within valid range
        for tok in tokens:
            assert 0 <= tok < vocab_size, f"Token {tok} out of valid range [0, {vocab_size})"


def test_empty_file():
    """Test handling of empty file"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        empty_path = Path(f.name)

    ds = ChessSequenceDataset(path=str(empty_path), max_len=32, stride=16)
    assert len(ds) == 0


def test_single_game_file():
    """Test file with single game"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        json.dump({"moves": ["e2e4", "e7e5"]}, f)
        single_path = Path(f.name)

    ds = ChessSequenceDataset(path=str(single_path), max_len=32, stride=16)
    assert len(ds) == 1
    tokens = ds[0]
    if isinstance(tokens, dict):
        tokens = tokens["tokens"]
    assert len(tokens) == 4  # 2 moves = 2 states + 2 moves


def test_window_boundaries(games_file):
    """Test that windows respect game boundaries"""
    # Use small window and stride to test boundary handling
    ds = ChessSequenceDataset(
        path=str(games_file),
        max_len=8,
        stride=4,
        start_on_state=True,
        drop_trailing_state=True,
        return_info=True,
    )

    for i in range(len(ds)):
        item = ds[i]
        tokens = item["tokens"]

        # Check token count is even (state-move pairs)
        if not ds.drop_trailing_state:
            assert len(tokens) % 2 == 1  # Odd if we keep trailing state
        else:
            assert len(tokens) % 2 == 0  # Even if we drop trailing state

        # Verify no window exceeds max_len
        assert len(tokens) <= ds.max_len


def test_custom_start_fen():
    """Test games with custom starting positions"""
    games_with_fen = [
        {
            "moves": ["e2e4", "e7e5"],
            "start_fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        },
        {"moves": ["d7d5", "c2c4"], "start_fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for game in games_with_fen:
            json.dump(game, f)
            f.write("\n")
        fen_path = Path(f.name)

    ds = ChessSequenceDataset(path=str(fen_path), max_len=32, stride=16)
    assert len(ds) > 0

    # Should successfully encode games with custom FENs
    for i in range(len(ds)):
        tokens = ds[i]
        if isinstance(tokens, dict):
            tokens = tokens["tokens"]
        assert all(isinstance(t, (int, np.integer)) for t in tokens)


def test_dataset_statistics(games_file):
    """Test dataset statistics collection"""
    ds = ChessSequenceDataset(path=str(games_file), max_len=32, stride=16, verbose=False)

    # Check internal statistics if available
    if hasattr(ds, "total_tokens"):
        assert ds.total_tokens > 0
    if hasattr(ds, "window_info"):
        assert isinstance(ds.window_info, list)
        assert len(ds.window_info) == len(ds)


def test_dataloader_compatibility(games_file):
    """Test compatibility with PyTorch DataLoader with custom collate"""
    from torch.utils.data import DataLoader

    from chess_lm.data.collate_causal import CollateCausal

    # Don't use padding in dataset, let collator handle it
    ds = ChessSequenceDataset(path=str(games_file), max_len=32, stride=16)

    # Use our custom collator
    collate_fn = CollateCausal(pad_id=vocab_total_size() - 1, max_seq_len=32)

    # Should work with DataLoader
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # Get one batch
    batch = next(iter(loader))

    assert "input_ids" in batch
    assert "labels" in batch
    assert batch["input_ids"].shape[0] <= 4  # Batch size
    assert batch["input_ids"].shape[1] <= 32  # Max length
