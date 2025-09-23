# chess_lm/tokenizer/encoding.py
from __future__ import annotations
import json
from typing import List, Dict, Any
from .moves import uci_to_id, id_to_uci, vocab_size
from .state_tokens import state_index, NUM_STATE_TOKENS

# We place state tokens AFTER the move vocab by offsetting.
def encode_state(st_idx: int) -> int:
    return vocab_size() + st_idx

def is_state_token(tok: int) -> bool:
    return tok >= vocab_size()

def encode_game(game_dict: Dict[str, Any]) -> List[int]:
    """
    game_dict: {
      "states": [{"stm_white":true, "castles":"KQkq", "ep_file":null}, ...],
      "moves":  ["e2e4", "e7e5", ...]  # same length as states
    }
    """
    toks: List[int] = []
    for st, mv in zip(game_dict["states"], game_dict["moves"]):
        st_idx = state_index(st["stm_white"], st["castles"], st["ep_file"])
        toks.append(encode_state(st_idx))
        toks.append(uci_to_id(mv))
    return toks

def vocab_total_size() -> int:
    return vocab_size() + NUM_STATE_TOKENS

def decode_moves_only(tokens: List[int]) -> List[str]:
    """Decode only the move tokens from a list, ignoring state tokens."""
    moves = []
    for tok in tokens:
        if not is_state_token(tok):
            moves.append(id_to_uci(tok))
    return moves