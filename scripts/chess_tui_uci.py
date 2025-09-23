#!/usr/bin/env python3
"""
Minimal terminal chess CLI (UCI-first).

Enter UCI moves directly:
  e2e4, e7e5, g1f3, b8c6, e1g1, a7a8q

Prefixes:
  san:...   play a SAN move (e.g., "san:Nf3", "san:O-O", "san:a8=Q")

Commands:
  help           Show this help
  show           Print board again
  moves          List legal moves (UCI)
  moves san      List legal moves (SAN)
  fen            Print FEN
  undo [n]       Undo last move(s)
  reset          Start a new game
  flip           Flip orientation (Black at bottom)
  save [file]    Save current board as SVG (default board.svg)
  quit / exit    Leave
"""
from __future__ import annotations
import sys
import shlex
from typing import List, Optional

import chess
import chess.svg

BANNER = "Chess TUI (UCI) — type UCI (e2e4, g1f3, e1g1, a7a8q) — `help` for commands."

def print_board(board: chess.Board, flip: bool = False) -> None:
    s = str(board if not flip else board.mirror())
    rows = s.splitlines()
    files = "abcdefgh" if not flip else "hgfedcba"
    rank_labels = list(range(8, 0, -1)) if not flip else list(range(1, 9))
    print()
    for i, row in enumerate(rows):
        print(f"{rank_labels[i]}  {row}")
    print(f"   {' '.join(files)}\n")

def list_legal_uci(board: chess.Board) -> List[str]:
    return [m.uci() for m in board.legal_moves]

def list_legal_san(board: chess.Board) -> List[str]:
    return [board.san(m) for m in board.legal_moves]

def save_svg(board: chess.Board, out_path: str = "board.svg", last: Optional[chess.Move] = None, flip: bool=False) -> str:
    svg = chess.svg.board(
        board=board,
        coordinates=True,
        orientation=chess.BLACK if flip else chess.WHITE,
        lastmove=last,
        arrows=[chess.svg.Arrow(last.from_square, last.to_square)] if last else None,
    )
    with open(out_path, "w") as f:
        f.write(svg)
    return out_path

def _print_cols(items: List[str], width: int = 90) -> None:
    line = ""
    for x in items:
        cell = f"{x}  "
        if len(line) + len(cell) > width:
            print(line.rstrip()); line = cell
        else:
            line += cell
    if line.strip():
        print(line.rstrip())

def repl():
    board = chess.Board()
    flip = False
    last_move: Optional[chess.Move] = None

    print(BANNER)
    print_board(board, flip)
    while True:
        stm = "White" if board.turn else "Black"
        try:
            line = input(f"[{stm} to move] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye."); return
        if not line:
            continue

        parts = shlex.split(line)
        cmd = parts[0].lower()

        if cmd in ("quit", "exit"):
            print("bye."); return

        if cmd == "help":
            print(__doc__); continue

        if cmd == "show":
            print_board(board, flip); continue

        if cmd == "moves":
            # "moves" or "moves san"
            mode = parts[1].lower() if len(parts) > 1 else "uci"
            if mode == "san":
                ms = list_legal_san(board)
            else:
                ms = list_legal_uci(board)
            if not ms:
                print("(no legal moves)")
            else:
                _print_cols(ms)
            continue

        if cmd == "fen":
            print(board.fen()); continue

        if cmd == "undo":
            n = 1
            if len(parts) > 1:
                try:
                    n = max(1, int(parts[1]))
                except ValueError:
                    print("usage: undo [n]"); continue
            for _ in range(min(n, len(board.move_stack))):
                board.pop()
            last_move = board.move_stack[-1] if board.move_stack else None
            print_board(board, flip); continue

        if cmd == "reset":
            board.reset(); last_move = None
            print_board(board, flip); continue

        if cmd == "flip":
            flip = not flip
            print_board(board, flip); continue

        if cmd == "save":
            out = parts[1] if len(parts) > 1 else "board.svg"
            path = save_svg(board, out, last_move, flip=flip)
            print(f"saved: {path}"); continue

        # Moves: UCI default, or SAN with "san:" prefix
        try:
            if cmd.startswith("san:"):
                san_text = line[4:] if line.lower().startswith("san:") else cmd[4:]
                mv = board.parse_san(san_text)
            else:
                # treat whole line as a UCI string (e.g. "e2e4", "a7a8q")
                uci = line
                mv = chess.Move.from_uci(uci)
                if mv not in board.legal_moves:
                    print(f"illegal in this position: {uci}")
                    continue
            board.push(mv)
            last_move = mv
            print_board(board, flip)
        except ValueError as e:
            print(f"could not parse move: {line} — {e}")
        except Exception as e:
            print(f"error: {e}")

def main():
    try:
        repl()
    except ImportError as e:
        if "chess" in str(e):
            print("python-chess is not installed. Try: pip install python-chess")
        else:
            raise

if __name__ == "__main__":
    main()
