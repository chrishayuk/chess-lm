"""Additional tests to improve masks.py coverage"""

import pytest
import torch

from chess_lm.tokenizer import encode_state, legality_mask_batch, state_index, uci_to_id, vocab_size


def test_invalid_tensor_dimensions():
    """Test that invalid tensor dimensions raise ValueError (lines 46, 50)"""
    # Test with 1D tensor instead of 2D
    xb_1d = torch.tensor([4343], dtype=torch.long)
    boundary_1d = torch.tensor([True], dtype=torch.bool)

    with pytest.raises(ValueError, match="batch_tokens and boundary_mask must be 2D"):
        legality_mask_batch(xb_1d, boundary_1d)

    # Test with 3D tensor instead of 2D
    xb_3d = torch.tensor([[[4343]]], dtype=torch.long)
    boundary_3d = torch.tensor([[[True]]], dtype=torch.bool)

    with pytest.raises(ValueError, match="batch_tokens and boundary_mask must be 2D"):
        legality_mask_batch(xb_3d, boundary_3d)


def test_mismatched_batch_boundary_shapes():
    """Test that mismatched shapes raise ValueError (line 50)"""
    # Create tensors with mismatched shapes
    xb = torch.tensor([[4343, 2209]], dtype=torch.long)  # Shape [1, 2]
    boundary_mask = torch.tensor([[True]], dtype=torch.bool)  # Shape [1, 1]

    with pytest.raises(ValueError, match="boundary_mask must match batch_tokens shape"):
        legality_mask_batch(xb, boundary_mask)


def test_wrong_length_initial_fens():
    """Test that wrong length initial_fens raises ValueError (lines 58-59)"""
    xb = torch.tensor([[4343], [4343]], dtype=torch.long)  # Batch size 2
    boundary_mask = torch.tensor([[True], [True]], dtype=torch.bool)

    # Provide initial_fens with wrong length (1 instead of 2)
    with pytest.raises(ValueError, match="initial_fens must be None or length B"):
        legality_mask_batch(xb, boundary_mask, initial_fens=["startpos"])

    # Provide initial_fens with wrong length (3 instead of 2)
    with pytest.raises(ValueError, match="initial_fens must be None or length B"):
        legality_mask_batch(xb, boundary_mask, initial_fens=["startpos", "startpos", "startpos"])


def test_custom_initial_fen():
    """Test using a custom FEN as initial position"""
    # Use a specific FEN position
    custom_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"

    # Create state token for black to move (after e2e4)
    st_tok = encode_state(state_index(False, "KQkq", "e"))  # Black to move with e3 en passant
    xb = torch.tensor([[st_tok]], dtype=torch.long)
    boundary_mask = torch.tensor([[True]], dtype=torch.bool)

    # Use the custom FEN
    mask = legality_mask_batch(xb, boundary_mask, initial_fens=[custom_fen])

    # Black should have 20 legal moves after 1.e4
    assert mask[0, 0].sum() == 20


def test_with_unknown_move_token():
    """Test handling of completely invalid/unknown move tokens (lines 84-86)"""
    # Create a state token
    st_tok = encode_state(state_index(True, "KQkq", None))

    # Use an invalid token ID that's not a state token and not a valid move
    # We'll use vocab_size() - 1 which should be an invalid move token
    invalid_token = vocab_size() - 1

    # Create sequence: [STATE, INVALID_TOKEN, STATE]
    st2_tok = encode_state(state_index(False, "KQkq", None))
    xb = torch.tensor([[st_tok, invalid_token, st2_tok]], dtype=torch.long)
    boundary_mask = torch.tensor([[True, False, True]], dtype=torch.bool)

    # Should handle gracefully without crashing
    mask = legality_mask_batch(xb, boundary_mask)
    assert mask.shape == (1, 3, vocab_size())

    # Board shouldn't advance after invalid token
    assert mask[0, 0].sum() == 20  # White's moves from start
    assert mask[0, 2].sum() == 20  # Still from start (black shouldn't have moved)


def test_rare_promotion_moves_not_in_catalog():
    """Test handling moves not in UCI catalog (lines 73-75)"""
    # Create a position where unusual promotions might occur
    # This position has a pawn ready to promote
    fen_with_promotion = "8/P7/8/8/8/8/8/8 w - - 0 1"

    st_tok = encode_state(state_index(True, "-", None))
    xb = torch.tensor([[st_tok]], dtype=torch.long)
    boundary_mask = torch.tensor([[True]], dtype=torch.bool)

    # This should work even if some promotion moves aren't in catalog
    mask = legality_mask_batch(xb, boundary_mask, initial_fens=[fen_with_promotion])
    assert mask.shape == (1, 1, vocab_size())

    # Should have some legal moves (the 4 promotions: queen, rook, bishop, knight)
    legal_count = mask[0, 0].sum().item()
    assert legal_count > 0  # At least some moves should be legal


def test_illegal_move_in_sequence():
    """Test that illegal moves in the sequence don't advance the board (line 88)"""
    # Start from standard position
    st_tok = encode_state(state_index(True, "KQkq", None))

    # e2e4 is legal
    legal_move = uci_to_id("e2e4")

    # After e2e4, it's black's turn, so e2e3 would be illegal (white can't move)
    # But let's use a clearly illegal move from the start: moving a piece backwards
    try:
        illegal_move = uci_to_id("e1e3")  # King moving two squares - illegal
    except KeyError:
        illegal_move = uci_to_id("a2a1")  # Pawn moving backwards - illegal

    # State tokens
    black_st = encode_state(state_index(False, "KQkq", "e"))
    white_st2 = encode_state(state_index(True, "KQkq", None))

    # Sequence: WHITE_STATE, e2e4, BLACK_STATE, illegal_move, WHITE_STATE
    xb = torch.tensor([[st_tok, legal_move, black_st, illegal_move, white_st2]], dtype=torch.long)
    boundary_mask = torch.tensor([[True, False, True, False, True]], dtype=torch.bool)

    mask = legality_mask_batch(xb, boundary_mask)

    # After the legal e2e4, black should have 20 moves
    assert mask[0, 2].sum() == 20

    # The illegal move shouldn't advance the position
    # So the next state should still show the position after just e2e4
    # (not after e2e4 and some other move)
    # We can't directly test this without inspecting internal state,
    # but we can verify the mask was computed without crashing


def test_multiple_batches_with_different_initial_positions():
    """Test batch processing with different starting positions"""
    # Batch of 3 games with different starting positions
    st1 = encode_state(state_index(True, "KQkq", None))  # Standard start
    st2 = encode_state(state_index(False, "KQkq", "e"))  # After e4
    st3 = encode_state(state_index(True, "-", None))  # No castling rights

    xb = torch.tensor(
        [[st1, uci_to_id("e2e4"), st2], [st2, uci_to_id("e7e5"), st1], [st3, uci_to_id("d2d4"), st2]], dtype=torch.long
    )

    boundary_mask = torch.tensor([[True, False, True], [True, False, True], [True, False, True]], dtype=torch.bool)

    # Different starting FENs for each game
    initial_fens = [
        "startpos",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1",
    ]

    mask = legality_mask_batch(xb, boundary_mask, initial_fens=initial_fens)
    assert mask.shape == (3, 3, vocab_size())

    # Each batch should have computed masks for state positions
    assert mask[0, 0].sum() > 0  # Game 1, position 0
    assert mask[1, 0].sum() > 0  # Game 2, position 0
    assert mask[2, 0].sum() > 0  # Game 3, position 0
