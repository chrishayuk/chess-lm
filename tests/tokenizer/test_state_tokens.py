from chess_lm.tokenizer import NUM_STATE_TOKENS, state_index


def test_state_index_basic():
    # Test basic state encoding
    idx = state_index(True, "KQkq", None)
    assert 0 <= idx < NUM_STATE_TOKENS

    idx = state_index(False, "-", "e")
    assert 0 <= idx < NUM_STATE_TOKENS


def test_state_index_all_combinations():
    # Test that all valid combinations produce unique indices
    indices = set()

    for stm in [True, False]:
        for castles in ["KQkq", "KQk", "KQ", "Kkq", "K", "Qkq", "kq", "k", "q", "-"]:
            for ep in [None, "a", "b", "c", "d", "e", "f", "g", "h"]:
                idx = state_index(stm, castles, ep)
                assert 0 <= idx < NUM_STATE_TOKENS
                indices.add(idx)

    # Check we generated unique indices
    assert len(indices) > 0


def test_invalid_castle_flags():
    # Test that invalid castle flags fall back to "-"
    # This tests line 16 in state_tokens.py
    idx_invalid = state_index(True, "XYZ", None)  # Invalid castle string
    idx_none = state_index(True, "-", None)  # Should be same as "-"
    assert idx_invalid == idx_none

    # Test another invalid case
    idx_invalid2 = state_index(False, "invalid", "e")
    idx_none2 = state_index(False, "-", "e")
    assert idx_invalid2 == idx_none2


def test_castle_string_normalization():
    # Test that castle strings are normalized (sorted)
    idx1 = state_index(True, "kqKQ", None)  # Unsorted
    idx2 = state_index(True, "KQkq", None)  # Sorted
    assert idx1 == idx2

    idx3 = state_index(True, "qkQK", None)  # Another unsorted variant
    assert idx3 == idx2
