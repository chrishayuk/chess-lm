#!/usr/bin/env python3
"""
Hello script for ChessSequenceDataset (dataset only).
"""
import argparse
from transformers import AutoTokenizer
from chess_lm.data.chess_sequence_dataset import ChessSequenceDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to JSONL file (games)")
    ap.add_argument("--vocab", default="out/vocab", help="HF tokenizer folder")
    ap.add_argument("--max-len", type=int, default=128, help="window length")
    ap.add_argument("--stride", type=int, default=None, help="hop between windows (default: max_len//2)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--lazy", action="store_true", help="use lazy loading mode")
    ap.add_argument("--show-n", type=int, default=2, help="how many windows to print")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.vocab)

    ds = ChessSequenceDataset(
        path=args.data,
        max_len=args.max_len,
        stride=args.stride if args.stride is not None else args.max_len // 2,
        seed=args.seed,
        start_on_state=True,
        drop_trailing_state=True,
        lazy_load=args.lazy,
        cache_dir=None,
        validate_tokens=True,
        pad_short_sequences=False,
        return_info=False,
        verbose=True,
    )

    # stats = ds.get_stats()
    print("\n=== Dataset info ===")
    print(f"Dataset loaded with {len(ds)} windows")
    # print(f"games     : {stats.total_games}")
    # print(f"windows   : {stats.total_windows}")
    # print(f"tokens    : {stats.total_tokens}")
    # print(f"avg len   : {stats.avg_game_length:.1f}")
    # print(f"len range : [{stats.min_game_length}, {stats.max_game_length}]")
    # print(f"vocab size: {stats.vocab_size}")

    # Peek a few windows
    print(f"\n=== First {args.show_n} window(s) ===")
    for i in range(min(args.show_n, len(ds))):
        ids = ds[i] if isinstance(ds[i], list) else ds[i]["tokens"]
        toks = tok.convert_ids_to_tokens(ids)
        print(f"[{i}] len={len(ids)}")
        print(" ids  :", ids[:min(32, len(ids))], "..." if len(ids) > 32 else "")
        print(" toks :", toks[:min(32, len(toks))], "..." if len(toks) > 32 else "")


if __name__ == "__main__":
    main()
