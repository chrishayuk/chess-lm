import chess
import torch

from chess_lm.tokenizer import UCI2ID, encode_state, legality_mask_batch, state_index, uci_to_id, vocab_size


def test_legality_mask_includes_all_legal_moves():
    b = chess.Board()  # start position

    # Build a minimal token sequence with one state token (predict next move)
    def statize(b: chess.Board):
        ep = b.ep_square
        ep_file = chess.square_file(ep) if ep is not None else None
        ep_file_char = "abcdefgh"[ep_file] if ep_file is not None else None
        castles = b.castling_xfen() or "-"
        return {"stm_white": b.turn, "castles": castles, "ep_file": ep_file_char}

    st = statize(b)
    st_tok = encode_state(state_index(st["stm_white"], st["castles"], st["ep_file"]))
    xb = torch.tensor([[st_tok]], dtype=torch.long)  # [B=1, T=1]
    boundary_mask = torch.tensor([[True]], dtype=torch.bool)  # state at t=0 => next is a move

    mask = legality_mask_batch(xb, boundary_mask)  # [1,1,V_moves]
    legal_ucis = [m.uci() for m in b.legal_moves]
    legal_ids = set(uci_to_id(u) for u in legal_ucis if u in UCI2ID)

    row = mask[0, 0].nonzero(as_tuple=False).flatten().tolist()
    got = set(row)
    # All legal IDs must be present
    missing = legal_ids - got
    assert not missing, f"Missing legal ids: {missing}"


def test_illegal_moves_masked_out():
    b = chess.Board()
    st = {"stm_white": b.turn, "castles": b.castling_xfen() or "-", "ep_file": None}
    st_tok = encode_state(state_index(st["stm_white"], st["castles"], st["ep_file"]))
    xb = torch.tensor([[st_tok]], dtype=torch.long)
    boundary_mask = torch.tensor([[True]], dtype=torch.bool)

    mask = legality_mask_batch(xb, boundary_mask)[0, 0]  # [V_moves]
    # Pick a move that's impossible in the initial position (e.g., "a2a1q" promotion)
    assert "a2a1q" in UCI2ID
    assert not mask[uci_to_id("a2a1q")]


def test_mask_with_illegal_move_tokens():
    """Test that illegal move tokens in the sequence are handled gracefully"""
    # Create a sequence with a state token followed by an illegal move token
    st = {"stm_white": True, "castles": "KQkq", "ep_file": None}
    st_tok = encode_state(state_index(st["stm_white"], st["castles"], st["ep_file"]))

    # Use an illegal move for the starting position (e.g., a pawn promotion)
    illegal_move_id = uci_to_id("a2a1q")  # This is illegal from start position

    # Create a sequence: [STATE, ILLEGAL_MOVE, STATE]
    st2_tok = encode_state(state_index(False, "KQkq", None))
    xb = torch.tensor([[st_tok, illegal_move_id, st2_tok]], dtype=torch.long)
    boundary_mask = torch.tensor([[True, False, True]], dtype=torch.bool)

    # Should not crash when processing the illegal move
    mask = legality_mask_batch(xb, boundary_mask)

    # The mask shape should be correct
    assert mask.shape == (1, 3, vocab_size())

    # First position should have legal moves from start
    assert mask[0, 0].sum() == 20  # 20 legal moves from start position

    # After illegal move, board state shouldn't advance, so next mask
    # should still be from starting position (but for black)
    assert mask[0, 2].sum() == 20  # 20 legal moves for black from start


def test_mask_handles_unknown_uci():
    """Test that mask handles moves not in the UCI catalog"""
    # Position where h7h8k would be legal: "8/7P/8/8/8/8/8/8 w - - 0 1"

    st = {"stm_white": True, "castles": "-", "ep_file": None}
    st_tok = encode_state(state_index(st["stm_white"], st["castles"], st["ep_file"]))
    xb = torch.tensor([[st_tok]], dtype=torch.long)
    boundary_mask = torch.tensor([[True]], dtype=torch.bool)

    # This should not crash even if some UCI moves aren't in our catalog
    mask = legality_mask_batch(xb, boundary_mask)
    assert mask.shape == (1, 1, vocab_size())
