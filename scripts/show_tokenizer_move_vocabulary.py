#!/usr/bin/env python3
"""
Show the complete move vocabulary from the existing tokenizer.

- Uses moves.py (UCI catalog already built).
- Prints vocabulary size, sample moves, and optionally dumps all to file.
"""

from __future__ import annotations
import argparse

from chess_lm.tokenizer import id_to_uci, vocab_size

def main():
    ap = argparse.ArgumentParser(description="Show move vocabulary from tokenizer.")
    ap.add_argument("--out", type=str, default=None,
                    help="Optional file to dump full vocabulary (id -> move).")
    ap.add_argument("--sample", type=int, default=50,
                    help="Number of sample moves to print (default: 50).")
    args = ap.parse_args()

    size = vocab_size()
    print("=== Move Vocabulary (from tokenizer) ===")
    print(f"Total moves in catalog: {size}\n")

    # Print sample
    print(f"Sample {args.sample} moves:")
    for i in range(min(args.sample, size)):
        print(f"{i:4d}  {id_to_uci(i)}")

    # Dump to file if requested
    if args.out:
        with open(args.out, "w") as f:
            for i in range(size):
                f.write(f"{i}\t{id_to_uci(i)}\n")
        print(f"\nFull vocabulary written to {args.out}")

if __name__ == "__main__":
    main()
