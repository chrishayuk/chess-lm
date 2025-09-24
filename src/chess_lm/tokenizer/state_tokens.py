## chess_lm/state_tokens.py
# Compact, lossy-but-useful state tokens interleaved before each move.
# Encodes: side to move (2), castling rights (16 combos), ep file bucket (9)
# => 2 * 16 * 9 = 288 tokens. We offset them above move vocab.
from __future__ import annotations

CASTLE_FLAGS = ["-", "K", "Q", "k", "q", "KQ", "Kk", "Kq", "Qk", "Qq", "kq", "KQk", "KQq", "Kkq", "Qkq", "KQkq"]
FILE2IDX = {f: i + 1 for i, f in enumerate("abcdefgh")}  # 1..8, 0==no ep
IDX2FILE = {v: k for k, v in FILE2IDX.items()}


def castle_to_idx(s: str) -> int:
    # s is e.g. "KQkq", "-" etc., normalize to one of 16
    s = "".join(sorted(s)) if s != "-" else "-"
    if s not in CASTLE_FLAGS:  # fallback
        s = "-"
    return CASTLE_FLAGS.index(s)


def ep_to_idx(file_char: str | None) -> int:
    return 0 if not file_char else FILE2IDX.get(file_char, 0)


def state_index(side_to_move_white: bool, castles: str, ep_file: str | None) -> int:
    stm = 0 if side_to_move_white else 1
    c = castle_to_idx(castles)
    e = ep_to_idx(ep_file)
    return (stm * 16 * 9) + (c * 9) + e  # 0..287


NUM_STATE_TOKENS = 288
