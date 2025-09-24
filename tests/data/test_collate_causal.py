"""Tests for collate_causal module"""

import torch

from chess_lm.data.collate_causal import CollateCausal
from chess_lm.tokenizer import is_state_token, vocab_total_size


def test_collate_causal_basic():
    """Test basic collation functionality"""
    collate = CollateCausal(pad_id=vocab_total_size())

    batch = [
        [4343, 100, 4400, 200],  # state, move, state, move
        [4343, 101],  # state, move
    ]

    result = collate(batch)

    assert "input_ids" in result
    assert "labels" in result
    assert "attention_mask" in result
    assert "loss_mask" in result

    # Check shapes - should be padded to longest sequence
    assert result["input_ids"].shape == (2, 4)
    assert result["labels"].shape == (2, 4)


def test_collate_causal_padding():
    """Test that shorter sequences are padded correctly"""
    pad_id = vocab_total_size()
    collate = CollateCausal(pad_id=pad_id)

    batch = [
        [4343, 100, 4400, 200],  # Length 4
        [4343, 100],  # Length 2
    ]

    result = collate(batch)

    # First sequence should be unchanged
    assert result["input_ids"][0].tolist() == [4343, 100, 4400, 200]

    # Second sequence should be padded
    assert result["input_ids"][1, :2].tolist() == [4343, 100]
    assert result["input_ids"][1, 2:].tolist() == [pad_id, pad_id]

    # Attention mask should reflect padding
    assert result["attention_mask"][0].tolist() == [True, True, True, True]
    assert result["attention_mask"][1].tolist() == [True, True, False, False]


def test_collate_causal_labels():
    """Test label generation with ignore index"""
    collate = CollateCausal(pad_id=vocab_total_size(), ignore_index=-100)

    batch = [
        [4343, 100, 4400, 200],
        [4343, 100],
    ]

    result = collate(batch)

    # Labels should match input_ids where attention_mask is True
    assert result["labels"][0].tolist() == [4343, 100, 4400, 200]

    # Padded positions should have ignore_index
    assert result["labels"][1, :2].tolist() == [4343, 100]
    assert all(result["labels"][1, 2:] == -100)


def test_collate_causal_truncation():
    """Test truncation when max_seq_len is set"""
    collate = CollateCausal(pad_id=vocab_total_size(), max_seq_len=4)

    batch = [
        [4343, 100, 4400, 200, 4343, 300, 4400, 400],  # Length 8, will be truncated
        [4343, 100],  # Length 2
    ]

    result = collate(batch)

    # Should be truncated to max_seq_len
    assert result["input_ids"].shape[1] == 4
    assert result["input_ids"][0].tolist() == [4343, 100, 4400, 200]


def test_collate_causal_boundary_mask():
    """Test boundary mask generation"""
    collate = CollateCausal(pad_id=vocab_total_size(), make_boundary_mask=True)

    batch = [
        [4343, 100, 4400, 200],  # state, move, state, move
    ]

    result = collate(batch)

    assert "boundary_mask" in result
    boundary = result["boundary_mask"][0]

    # boundary_mask should be True for state tokens
    assert boundary[0] == True  # First state
    assert boundary[1] == False  # First move
    assert boundary[2] == True  # Second state
    assert boundary[3] == False  # Second move


def test_collate_causal_only_moves_masking():
    """Test 'only_moves' masking mode"""
    collate = CollateCausal(pad_id=vocab_total_size(), mask_mode="only_moves")

    batch = [
        [4343, 100, 4400, 200],  # state, move, state, move
    ]

    result = collate(batch)
    loss_mask = result["loss_mask"][0]

    # Should only train on positions where we predict moves (after states)
    assert loss_mask[0] == False  # Position 0 (state)
    assert loss_mask[1] == True  # Position 1 (move after state)
    assert loss_mask[2] == False  # Position 2 (state after move)
    assert loss_mask[3] == True  # Position 3 (move after state)


def test_collate_causal_empty_batch():
    """Test handling of empty batch"""
    collate = CollateCausal(pad_id=vocab_total_size())

    batch = []
    result = collate(batch)

    assert result["input_ids"].shape == (0, 0)
    assert result["labels"].shape == (0, 0)
    assert result["attention_mask"].shape == (0, 0)


def test_collate_causal_single_sequence():
    """Test collating a single sequence"""
    collate = CollateCausal(pad_id=vocab_total_size())

    batch = [[4343, 100, 4400, 200]]

    result = collate(batch)

    assert result["input_ids"].shape == (1, 4)
    assert result["input_ids"][0].tolist() == [4343, 100, 4400, 200]


def test_collate_causal_all_same_length():
    """Test collating sequences of the same length"""
    collate = CollateCausal(pad_id=vocab_total_size())

    batch = [
        [4343, 100, 4400, 200],
        [4343, 101, 4401, 201],
        [4343, 102, 4402, 202],
    ]

    result = collate(batch)

    # No padding should be needed
    assert result["input_ids"].shape == (3, 4)
    # Check sequences preserved
    for i in range(3):
        assert result["input_ids"][i].tolist() == batch[i]


def test_collate_causal_dtype():
    """Test that output tensors have correct dtype"""
    collate = CollateCausal(pad_id=vocab_total_size())

    batch = [[4343, 100, 4400, 200]]

    result = collate(batch)

    assert result["input_ids"].dtype == torch.long
    assert result["labels"].dtype == torch.long
    assert result["attention_mask"].dtype == torch.bool
    assert result["loss_mask"].dtype == torch.bool


def test_collate_causal_device():
    """Test tensor device placement"""
    collate = CollateCausal(pad_id=vocab_total_size())

    batch = [[4343, 100, 4400, 200]]

    result = collate(batch)

    # Should be on CPU by default
    assert result["input_ids"].device.type == "cpu"
    assert result["labels"].device.type == "cpu"


def test_collate_causal_mask_sim_mode():
    """Test 'mask_sim' masking mode (for simulation tokens)"""
    sim_begin = vocab_total_size() + 100
    sim_end = vocab_total_size() + 101

    collate = CollateCausal(
        pad_id=vocab_total_size(), mask_mode="mask_sim", sim_begin_id=sim_begin, sim_end_id=sim_end
    )

    # Sequence with simulation tokens
    batch = [
        [4343, 100, sim_begin, 200, 300, sim_end, 4400, 200],
    ]

    result = collate(batch)
    loss_mask = result["loss_mask"][0]

    # Should mask tokens inside sim_begin...sim_end
    assert loss_mask[0] == True  # Before sim_begin
    assert loss_mask[1] == True  # Before sim_begin
    assert loss_mask[2] == False  # sim_begin itself
    assert loss_mask[3] == False  # Inside simulation
    assert loss_mask[4] == False  # Inside simulation
    assert loss_mask[5] == True  # sim_end (not masked)
    assert loss_mask[6] == True  # After sim_end
    assert loss_mask[7] == True  # After sim_end


def test_collate_causal_large_batch():
    """Test collating a large batch"""
    collate = CollateCausal(pad_id=vocab_total_size())

    # Create a batch of 64 sequences with varying lengths
    batch = []
    for i in range(64):
        length = 10 + (i % 20)  # Varying lengths from 10 to 29
        tokens = []
        for j in range(length):
            if j % 2 == 0:
                tokens.append(4343 + (j % 100))  # State tokens
            else:
                tokens.append(100 + (j % 100))  # Move tokens
        batch.append(tokens)

    result = collate(batch)

    assert result["input_ids"].shape[0] == 64
    assert result["input_ids"].shape[1] == 29  # Max length in batch
    assert result["labels"].shape == result["input_ids"].shape
    assert result["attention_mask"].shape == result["input_ids"].shape


def test_collate_causal_no_boundary_mask():
    """Test disabling boundary mask generation"""
    collate = CollateCausal(pad_id=vocab_total_size(), make_boundary_mask=False)

    batch = [[4343, 100, 4400, 200]]

    result = collate(batch)

    # boundary_mask should not be in result
    assert "boundary_mask" not in result


def test_collate_causal_custom_ignore_index():
    """Test using custom ignore index"""
    collate = CollateCausal(pad_id=vocab_total_size(), ignore_index=-999)

    batch = [
        [4343, 100],
        [4343, 100, 4400, 200],
    ]

    result = collate(batch)

    # Padded positions should use custom ignore index
    assert all(result["labels"][0, 2:] == -999)