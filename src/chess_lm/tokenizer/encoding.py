# chess_lm/tokenizer/encoding.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import chess

from .moves import id_to_uci, uci_to_id, vocab_size
from .state_tokens import NUM_STATE_TOKENS, state_index

# --- ID helpers --------------------------------------------------------------


def encode_state(st_idx: int) -> int:
    """Map a state index into the global vocab space (after moves)."""
    return vocab_size() + st_idx


def is_state_token(tok: int) -> bool:
    """True iff token id belongs to the state-token range."""
    return tok >= vocab_size()


def vocab_total_size() -> int:
    """Total vocab size (moves + state tokens)."""
    return vocab_size() + NUM_STATE_TOKENS


# --- State derivation --------------------------------------------------------


def _state_from_board(board: chess.Board) -> Dict[str, Any]:
    """Derive compact state info from a board."""
    stm_white: bool = board.turn
    castles: str = board.castling_xfen() or "-"
    ep_file: Optional[str] = None
    if board.ep_square is not None:
        file_idx = chess.square_file(board.ep_square)
        ep_file = "abcdefgh"[file_idx]
    return {"stm_white": stm_white, "castles": castles, "ep_file": ep_file}


def _moves_list(game_dict: Dict[str, Any]) -> List[str]:
    """Extract a list of UCIs from a game dict."""
    if "moves" in game_dict and isinstance(game_dict["moves"], list):
        return game_dict["moves"]
    if "moves_uci" in game_dict and isinstance(game_dict["moves_uci"], list):
        return game_dict["moves_uci"]
    raise KeyError("encode_game: expected 'moves' or 'moves_uci' list in record")


# --- Public API --------------------------------------------------------------


def encode_game(game_dict: Dict[str, Any]) -> List[int]:
    """
    Encode a game into token ids interleaved per ply: [STATE, MOVE, STATE, MOVE, ...].

    Input schema:
        {
          "moves": ["e2e4","e7e5", ...],   # or "moves_uci"
          "start_fen": "startpos" | FEN    # optional (default "startpos")
        }
    """
    toks: List[int] = []
    moves = _moves_list(game_dict)
    start_fen = game_dict.get("start_fen", "startpos")
    board = chess.Board() if start_fen == "startpos" else chess.Board(start_fen)

    for u in moves:
        # state BEFORE the move
        st = _state_from_board(board)
        st_idx = state_index(st["stm_white"], st["castles"], st["ep_file"])
        toks.append(encode_state(st_idx))

        # move
        mv = chess.Move.from_uci(u)
        if mv not in board.legal_moves:
            raise ValueError(f"Illegal move '{u}' for position: {board.fen()}")
        toks.append(uci_to_id(u))
        board.push(mv)

    return toks  # we intentionally do NOT append the terminal state


def decode_moves_only(tokens: List[int]) -> List[str]:
    """Decode only the move tokens from a list, ignoring state tokens."""
    return [id_to_uci(t) for t in tokens if not is_state_token(t)]
