import chess
import pytest
from chess_lm.tokenizer import (
    vocab_size,
    state_index,
    NUM_STATE_TOKENS,
    encode_state,
    is_state_token,
    encode_game,
    vocab_total_size,
    decode_moves_only,
    uci_to_id
)


def test_state_token_range():
    V_moves = vocab_size()
    st_idx = state_index(True, "KQkq", None)
    tok = encode_state(st_idx)
    assert tok >= V_moves
    assert is_state_token(tok)
    # last state token also in range
    last = encode_state(NUM_STATE_TOKENS - 1)
    assert last < vocab_total_size()


def test_encode_game_interleaves():
    # Tiny one-game example from startpos: 1. e4 e5
    def statize(b: chess.Board):
        ep = b.ep_square
        ep_file = chess.square_file(ep) if ep is not None else None
        ep_file_char = "abcdefgh"[ep_file] if ep_file is not None else None
        castles = b.castling_xfen() or "-"
        return {"stm_white": b.turn, "castles": castles, "ep_file": ep_file_char}

    b = chess.Board()
    states, moves = [], []
    for mv in [chess.Move.from_uci("e2e4"), chess.Move.from_uci("e7e5")]:
        states.append(statize(b))
        moves.append(mv.uci())
        b.push(mv)

    toks = encode_game({"states": states, "moves": moves})
    # Expect [STATE, MOVE, STATE, MOVE]
    assert len(toks) == 4
    assert is_state_token(toks[0]) and not is_state_token(toks[1])
    assert is_state_token(toks[2]) and not is_state_token(toks[3])


def test_decode_moves_only():
    # Create a sequence with mixed state and move tokens
    move1_id = uci_to_id("e2e4")
    move2_id = uci_to_id("e7e5")
    move3_id = uci_to_id("g1f3")
    
    # Create state tokens (any valid state index)
    state_tok1 = encode_state(0)
    state_tok2 = encode_state(10)
    
    # Mix state and move tokens
    tokens = [state_tok1, move1_id, state_tok2, move2_id, move3_id]
    
    # decode_moves_only should only return the moves, ignoring state tokens
    decoded = decode_moves_only(tokens)
    assert decoded == ["e2e4", "e7e5", "g1f3"]
    
    # Test with only state tokens
    state_only = [state_tok1, state_tok2]
    assert decode_moves_only(state_only) == []
    
    # Test with only move tokens
    moves_only = [move1_id, move2_id, move3_id]
    assert decode_moves_only(moves_only) == ["e2e4", "e7e5", "g1f3"]