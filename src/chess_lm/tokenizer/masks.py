# chess_lm/tokenizer/masks.py
from __future__ import annotations

from typing import List, Optional

import chess
import torch

from .encoding import is_state_token
from .moves import id_to_uci, uci_to_id, vocab_size


@torch.no_grad()
def legality_mask_batch(
    batch_tokens: torch.Tensor,
    boundary_mask: torch.Tensor,
    initial_fens: Optional[List[str]] = None,
) -> torch.Tensor:
    """
    Build a per-position legality mask for MOVE predictions.

    Clean, single-path version:
      - We assume each sequence `b` starts at a known chess position, given by
        `initial_fens[b]` (or "startpos" if not provided).
      - We step through the sequence once, mutating a `chess.Board` as we go:
          * When boundary_mask[b, t] == True (i.e., CURRENT token is a STATE),
            we compute the legal moves for that position and mark them True
            in the output mask at [b, t, move_id].
          * When the CURRENT token is a MOVE token, we push it on the board
            if legal; otherwise we skip (early-training noise/tiny corruptions).

    Args:
        batch_tokens: [B, T] int64, interleaved STATE and MOVE ids.
        boundary_mask: [B, T] bool, True where CURRENT token is a STATE
                       (i.e., the next prediction is a MOVE).
        initial_fens: optional list of length B. Each entry is:
                        - "startpos" or omitted → standard initial position
                        - a FEN string → exact board for the first token

    Returns:
        mask: [B, T, V] bool where V == move_vocab_size.
              For state positions t (boundary_mask True), mask[b, t, mid]=True
              iff that move id is legal in that position. All other entries False.
    """
    if batch_tokens.ndim != 2 or boundary_mask.ndim != 2:
        raise ValueError("batch_tokens and boundary_mask must be 2D [B, T]")

    B, T = batch_tokens.shape
    if boundary_mask.shape != (B, T):
        raise ValueError("boundary_mask must match batch_tokens shape [B, T]")

    V = vocab_size()
    out = torch.zeros((B, T, V), dtype=torch.bool, device=batch_tokens.device)

    # Normalize initial FENs
    if initial_fens is None:
        initial_fens = ["startpos"] * B
    elif len(initial_fens) != B:
        raise ValueError("initial_fens must be None or length B")

    for b in range(B):
        fen = initial_fens[b]
        board = chess.Board() if fen == "startpos" else chess.Board(fen)

        # Single pass over timesteps; board stays in sync with tokens.
        for t in range(T):
            # If CURRENT token is a STATE, emit legal mask for predicting the next MOVE.
            if boundary_mask[b, t]:
                legal_ids = []
                for m in board.legal_moves:
                    try:
                        legal_ids.append(uci_to_id(m.uci()))
                    except KeyError:
                        # Move not present in the catalog (unlikely with standard chess); ignore.
                        pass
                if legal_ids:
                    out[b, t, legal_ids] = True

            # If CURRENT token is a MOVE, try to apply it to advance the board.
            tok = int(batch_tokens[b, t].item())
            if not is_state_token(tok):
                try:
                    mv = chess.Move.from_uci(id_to_uci(tok))
                except Exception:
                    # Unknown token → ignore
                    continue
                if mv in board.legal_moves:
                    board.push(mv)
                # If illegal (e.g., corrupted data), we simply don't push to avoid divergence.

    return out
