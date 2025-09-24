#!/usr/bin/env python3
"""
Minimal terminal chess CLI.

- Enter SAN moves:  e4, Nf3, O-O, a8=Q, exd5+, Qh5#
- Or UCI with prefix: uci:e2e4, uci:g1f3
- Commands:
    help           Show this help
    show           Print board again
    moves          List legal moves (SAN)
    fen            Print FEN
    undo [n]       Undo last move(s)
    reset          Start a new game
    flip           Flip orientation (Black at bottom)
    save [file]    Save current board as SVG (default board.svg)
    quit / exit    Leave

Tips:
- Checks/mates in SAN (+/#) are accepted.
- Promotions in SAN: a8=Q (or a8=Q+ for check).
- Castling in SAN: O-O, O-O-O
"""
from __future__ import annotations
import shlex
from typing import List, Optional

import chess
import chess.svg

BANNER = "Chess TUI — enter SAN (e.g., e4, Nf3, O-O) or `uci:...` — type `help` for commands."

def print_board(board: chess.Board, flip: bool = False) -> None:
    # Pretty ASCII with file/rank labels
    s = str(board if not flip else board.mirror())
    rows = s.splitlines()
    if flip:
        # when mirrored, ranks/files swap; relabel to actual bottom view
        files = "hgfedcba"
        rank_start = 1
    else:
        files = "abcdefgh"
        rank_start = 8
    print()
    for i, row in enumerate(rows):
        rank = rank_start - i if not flip else rank_start + i
        print(f"{rank}  {row}")
    print(f"   {' '.join(files)}\n")

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

def repl():
    board = chess.Board()
    flip = False
    last_move: Optional[chess.Move] = None

    print(BANNER)
    print_board(board, flip)
    while True:
        stm = "White" if board.turn else "Black"
        prompt = f"[{stm} to move] > "
        try:
            line = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            return
        if not line:
            continue

        # Commands
        parts = shlex.split(line)
        cmd = parts[0].lower()

        if cmd in ("quit", "exit"):
            print("bye.")
            return

        if cmd == "help":
            print(__doc__)
            continue

        if cmd == "show":
            print_board(board, flip)
            continue

        if cmd == "moves":
            san_moves = list_legal_san(board)
            if not san_moves:
                print("(no legal moves)")
            else:
                # compact columns
                width = 80
                line = ""
                for mv in san_moves:
                    cell = f"{mv}  "
                    if len(line) + len(cell) > width:
                        print(line.rstrip())
                        line = cell
                    else:
                        line += cell
                if line.strip():
                    print(line.rstrip())
            continue

        if cmd == "fen":
            print(board.fen())
            continue

        if cmd == "undo":
            n = 1
            if len(parts) > 1:
                try:
                    n = max(1, int(parts[1]))
                except ValueError:
                    print("usage: undo [n]")
                    continue
            for _ in range(min(n, len(board.move_stack))):
                board.pop()
            last_move = board.move_stack[-1] if board.move_stack else None
            print_board(board, flip)
            continue

        if cmd == "reset":
            board.reset()
            last_move = None
            print_board(board, flip)
            continue

        if cmd == "flip":
            flip = not flip
            print_board(board, flip)
            continue

        if cmd == "save":
            out = "board.svg"
            if len(parts) > 1:
                out = parts[1]
            path = save_svg(board, out, last_move, flip=flip)
            print(f"saved: {path}")
            continue

        # Moves: try UCI with prefix, else SAN
        move_text = line
        try:
            if move_text.lower().startswith("uci:"):
                uci = move_text[4:]
                mv = chess.Move.from_uci(uci)
                if mv not in board.legal_moves:
                    print(f"illegal in this position: {uci}")
                    continue
                board.push(mv)
                last_move = mv
                print_board(board, flip)
            else:
                # SAN
                mv = board.parse_san(move_text)
                board.push(mv)
                last_move = mv
                print_board(board, flip)
        except ValueError as e:
            print(f"could not parse move: {move_text} — {e}")
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
