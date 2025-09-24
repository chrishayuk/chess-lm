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

    _ = AutoTokenizer.from_pretrained(args.vocab)  # Validate vocab exists
    ds = ChessSequenceDataset(
        path=args.data,
        max_len=args.max_len,
        stride=args.stride if args.stride is not None else args.max_len,
        start_on_state=True,
        drop_trailing_state=True,
        lazy_load=False,
        validate_tokens=True,
        pad_short_sequences=False,
        return_info=True,   # so we can see start_pos/game_id
        verbose=False,
    )

    def check_alternation(seq):
        for i, t in enumerate(seq):
            want_state = (i % 2 == 0)
            if want_state != is_state_token(t):
                return f"bad alternation at i={i}"
        return "ok"

    print(f"windows: {len(ds)}")
    to_check = min(args.check_n, len(ds))
    for i in range(to_check):
        item = ds[i]
        ids = item["tokens"]
        alt = check_alternation(ids)

        info = ds.window_info[i] if hasattr(ds, 'window_info') else {}
        start_pos = info.get("start_pos", 0)

        verdict = alt
        legality_note = "skipped"
        if alt == "ok" and start_pos == 0:
            # Only verify legality when the window starts from the beginning of its game.
            uci = decode_moves_only(ids)
            b = chess.Board()
            ok = True
            for u in uci:
                m = chess.Move.from_uci(u)
                if m not in b.legal_moves:
                    ok = False
                    break
                b.push(m)
            legality_note = "ok" if ok else "illegal"
        print(f"[{i}] alternation={alt} | legality={legality_note} | start_pos={start_pos}")

if __name__ == "__main__":
    main()

