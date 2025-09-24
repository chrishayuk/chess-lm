#!/usr/bin/env python3
# scripts/check_dataset_window_sanity.py
from __future__ import annotations
import argparse
import chess
from transformers import AutoTokenizer

from chess_lm.data.chess_sequence_dataset import ChessSequenceDataset
from chess_lm.tokenizer.encoding import is_state_token, decode_moves_only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--vocab", default="out/vocab")
    ap.add_argument("--max-len", type=int, default=100)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--check-n", type=int, default=10, help="windows to check")
    args = ap.parse_args()

    # Validate vocab exists / loads
    _ = AutoTokenizer.from_pretrained(args.vocab)

    # Build dataset with initial FENs per window
    ds = ChessSequenceDataset(
        path=args.data,
        max_len=args.max_len,
        stride=args.stride if args.stride is not None else args.max_len,
        start_on_state=True,
        drop_trailing_state=True,
        lazy_load=False,
        validate_tokens=True,
        pad_short_sequences=False,
        return_info=True,       # so we can see metadata
        emit_initial_fen=True,  # <<— crucial for legality checks on mid-game windows
        verbose=False,
    )

    def check_alternation(seq):
        for i, t in enumerate(seq):
            want_state = (i % 2 == 0)
            if want_state != is_state_token(t):
                return f"bad alternation at i={i}"
        return "ok"

    def check_legality(ids, initial_fen: str | None) -> str:
        """Return 'ok' if the sequence of moves is legal from the given starting FEN, else 'illegal'."""
        try:
            b = chess.Board() if not initial_fen or initial_fen == "startpos" else chess.Board(initial_fen)
        except Exception:
            return "illegal"  # malformed fen
        for u in decode_moves_only(ids):
            m = chess.Move.from_uci(u)
            if m not in b.legal_moves:
                return "illegal"
            b.push(m)
        return "ok"

    print(f"windows: {len(ds)}")
    to_check = min(args.check_n, len(ds))

    for i in range(to_check):
        item = ds[i]  # dict with 'tokens' and metadata
        ids = item["tokens"]
        alt = check_alternation(ids)

        # Prefer the window's initial_fen, fall back to startpos if absent
        initial_fen = item.get("initial_fen", "startpos")

        # Pull start_pos mainly for display
        start_pos = item.get("start_pos", 0)

        legality_note = "skipped"
        if alt == "ok":
            legality_note = check_legality(ids, initial_fen)

        print(f"[{i}] alternation={alt} | legality={legality_note} | start_pos={start_pos}")


if __name__ == "__main__":
    main()
