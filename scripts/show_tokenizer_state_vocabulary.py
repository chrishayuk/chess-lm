#!/usr/bin/env python3
"""
Show the complete set of STATE tokens.

Each state token encodes:
  - Side to move (White/Black)
  - Castling rights (KQkq subset or none)
  - En passant file (a–h or none)

IDs start at vocab_size() and run for NUM_STATE_TOKENS entries.
"""

from __future__ import annotations
from chess_lm.tokenizer import (
    vocab_size,
    NUM_STATE_TOKENS,
    encode_state,
    state_index,
)

FILES = "abcdefgh"

def all_state_combos():
    for stm_white in (True, False):
        for castles in [
            "", "K", "Q", "KQ", "k", "q", "kq",
            "Kk", "Kq", "Qk", "Qq", "KQk", "KQq", "Kkq", "Qkq", "KQkq"
        ]:
            for ep_file in [None] + list(FILES):
                yield stm_white, castles if castles else "-", ep_file

def main():
    mv_size = vocab_size()
    print("=== State Token Vocabulary ===")
    print(f"Move vocab size     : {mv_size}")
    print(f"State token count   : {NUM_STATE_TOKENS}")
    print(f"State token id span : {mv_size} .. {mv_size+NUM_STATE_TOKENS-1}\n")

    print(f"{'ID':>5}  {'Side':<5}  {'Castles':<6}  {'EP':<2}")
    print("-" * 28)

    local = 0
    for stm_white, castles, ep_file in all_state_combos():
        idx = state_index(stm_white, castles, ep_file)
        global_id = encode_state(idx)
        side = "W" if stm_white else "B"
        print(f"{global_id:5d}  {side:<5}  {castles:<6}  {ep_file or '-':<2}")
        local += 1

if __name__ == "__main__":
    main()
