# chess_lm/tokenizer/masks.py
from __future__ import annotations
import chess
import torch
from typing import List, Dict, Any
from .moves import uci_to_id, vocab_size
from .encoding import is_state_token
from .state_tokens import NUM_STATE_TOKENS

@torch.no_grad()
def legality_mask_batch(batch_tokens: torch.Tensor,
                        boundary_mask: torch.Tensor,
                        states_meta: List[List[Dict[str,Any]]]) -> torch.Tensor:
    """
    batch_tokens: [B, T] integer tokens (state & move tokens interleaved)
    boundary_mask: [B, T] bool True at positions where a MOVE will be predicted next
                   (i.e., positions that are state tokens)
    states_meta: for each sequence step, the precise FEN-ish info required to
                 rebuild a board up to that ply (we store during dataset build)
    returns: mask [B, T, V] where V == move_vocab, True=legal, False=illegal
    """
    B, T = batch_tokens.shape
    V = vocab_size()
    out = torch.zeros((B, T, V), dtype=torch.bool, device=batch_tokens.device)

    for b in range(B):
        # We track board by replaying from start; but we also stored exact states
        board = chess.Board()  # startpos
        # We'll step through; when we hit a state token at t, we compute legal set.
        for t in range(T):
            tok = batch_tokens[b, t].item()
            if is_state_token(tok):
                # Build legal moves from the *current* board:
                legal = []
                for m in board.legal_moves:
                    uci = m.uci()
                    try:
                        mid = uci_to_id(uci)
                        legal.append(mid)
                    except KeyError:
                        # Move not in our catalog (shouldn't happen with standard chess)
                        pass
                if legal:
                    out[b, t, legal] = True
            else:
                # It's a move token; push it if legal to advance board
                # Some tokens in early training may be illegal—guard it.
                try:
                    # board.parse_uci will raise if illegal
                    mv = chess.Move.from_uci(_id_to_uci(tok))  # helper; see below
                    if mv in board.legal_moves:
                        board.push(mv)
                except Exception:
                    # If illegal, skip push to avoid divergence; next state token mask will be broader
                    pass
    return out

# small helper (we import lazily to avoid cycle)
from .moves import id_to_uci as _id_to_uci
