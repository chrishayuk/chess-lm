#!/usr/bin/env python3
"""
Generate synthetic chess games (random legal self-play with a mild bias) to JSONL.

Each line:
  {"id": "g000001", "moves": ["e2e4","e7e5","g1f3", ...]}

Usage:
  uv run python scripts/generate_synthetic_games.py --out data/train_synth.jsonl --games 200
"""

from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path

import chess


def pick_move(board: chess.Board, rng: random.Random, bias: str = "light") -> chess.Move | None:
    """Pick a legal move with optional light heuristics."""
    moves = list(board.legal_moves)
    if not moves:
        return None
    if bias == "none":
        return rng.choice(moves)

    # Light bias: prefer captures, checks, and center pushes a bit.
    scored = []
    for m in moves:
        s = 1.0
        if board.is_capture(m):
            s += 0.9
        board.push(m)
        if board.is_check():
            s += 0.5
        to = m.to_square
        file = chess.square_file(to)
        rank = chess.square_rank(to)
        if file in (3, 4):  # center files d/e
            s += 0.2
        if rank in (3, 4):  # center ranks 4/5 (0-indexed)
            s += 0.2
        board.pop()
        scored.append((s, m))

    total = sum(s for s, _ in scored)
    r = rng.random() * total
    acc = 0.0
    for s, m in scored:
        acc += s
        if r <= acc:
            return m
    return moves[-1]


def play_one(rng: random.Random, min_plies: int, max_plies: int, bias: str) -> list[str]:
    """Play a single synthetic game and return UCI moves."""
    target_len = rng.randint(min_plies, max_plies)
    board = chess.Board()
    out: list[str] = []
    for _ in range(target_len):
        mv = pick_move(board, rng, bias=bias)
        if mv is None:
            break
        out.append(mv.uci())
        board.push(mv)
        if board.is_game_over():
            break
    return out


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic chess games to JSONL.")
    ap.add_argument("--out", required=True, help="Output JSONL file path")
    ap.add_argument("--games", type=int, default=100, help="Number of games to generate")
    ap.add_argument("--min-plies", type=int, default=12, help="Minimum plies per game (target)")
    ap.add_argument("--max-plies", type=int, default=80, help="Maximum plies per game (target)")
    ap.add_argument("--bias", choices=["light", "none"], default="light", help="Move selection bias")
    ap.add_argument("--seed", type=int, default=1337, help="RNG seed")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)  # ensure parent directory exists

    rng = random.Random(args.seed)
    n_written = 0

    with out_path.open("w", encoding="utf-8") as f:
        for i in range(args.games):
            moves = play_one(rng, args.min_plies, args.max_plies, args.bias)
            if len(moves) < max(1, args.min_plies // 2):
                # Skip extremely short or immediately-terminated games
                continue
            rec = {"id": f"g{i:06d}", "moves": moves}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Wrote {n_written} games → {out_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
