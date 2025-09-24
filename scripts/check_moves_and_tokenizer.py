#!/usr/bin/env python
import sys
import chess
from chess_lm.tokenizer import (
    UCI_CATALOG, uci_to_id, id_to_uci, vocab_size,
    encode_state, is_state_token, vocab_total_size,
    state_index, NUM_STATE_TOKENS,
    legality_mask_batch
)

def ok(cond, msg):
    print(("✅ " if cond else "❌ ") + msg)
    if not cond:
        sys.exit(1)

def main():
    print("=== Move catalog ===")
    ok(len(UCI_CATALOG) == len(set(UCI_CATALOG)), "catalog unique")
    ok(vocab_size() == len(UCI_CATALOG), "vocab_size matches catalog length")

    # Roundtrip spot checks
    for mv in ["e2e4", "e7e5", "e1g1", "e8c8", "a7a8q", "b7a8n"]:
        ok(id_to_uci(uci_to_id(mv)) == mv, f"roundtrip: {mv}")

    print("\n=== State tokens ===")
    V_moves = vocab_size()
    idx = state_index(True, "KQkq", None)
    st_tok = encode_state(idx)
    ok(st_tok >= V_moves, "state token offset above move vocab")
    ok(is_state_token(st_tok), "is_state_token works")
    ok(vocab_total_size() == V_moves + NUM_STATE_TOKENS, "total vocab size matches")

    print("\n=== Legality mask @ startpos ===")
    import torch
    b = chess.Board()
    ep = b.ep_square
    ep_file_char = None if ep is None else "abcdefgh"[chess.square_file(ep)]
    idx = state_index(b.turn, b.castling_xfen() or "-", ep_file_char)
    tok = encode_state(idx)
    xb = torch.tensor([[tok]], dtype=torch.long)
    boundary = torch.tensor([[True]], dtype=torch.bool)
    mask = legality_mask_batch(xb, boundary, states_meta=None)[0,0]  # [V_moves]

    legal = {uci_to_id(m.uci()) for m in b.legal_moves if m.uci() in UCI_CATALOG}
    got = set(mask.nonzero(as_tuple=False).flatten().tolist())
    ok(legal.issubset(got), f"all legal moves included ({len(legal)} moves)")

    # quick illegal probe
    bad = "a2a1q"
    if bad in UCI_CATALOG:
        ok(mask[uci_to_id(bad)].item() == 0, f"illegal masked: {bad}")

    print("\nAll checks passed.")

if __name__ == "__main__":
    main()
