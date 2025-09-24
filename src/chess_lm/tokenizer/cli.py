#!/usr/bin/env python3
# chess_lm/tokenizer/cli.py
"""
Tokenizer CLI (moves-first, optional state inspection)

Subcommands:
  info
  encode-seq --seq "e2e4 e7e5 g1f3" [--print-table|--json]
  decode-moves --ids "123 456,789" [--json]
  show-state --fen FEN [--json]
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

import chess

from . import (
    NUM_STATE_TOKENS,
    decode_moves_only,
    encode_state,
    id_to_uci,
    is_state_token,
    state_index,
    uci_to_id,
    vocab_size,
    vocab_total_size,
)


def _statize(b: chess.Board) -> Dict[str, Any]:
    ep = b.ep_square
    ep_file = None if ep is None else "abcdefgh"[chess.square_file(ep)]
    return {
        "stm_white": b.turn,
        "castles": b.castling_xfen() or "-",
        "ep_file": ep_file,
    }


def _encode_from_seq(seq: str) -> List[int]:
    toks: List[int] = []
    b = chess.Board()
    for u in seq.strip().split():
        st = _statize(b)
        st_idx = state_index(st["stm_white"], st["castles"], st["ep_file"])
        toks.append(encode_state(st_idx))
        move = chess.Move.from_uci(u)
        if move not in b.legal_moves:
            raise SystemExit(f"[encode-seq] illegal move '{u}' for position: {b.fen()}")
        toks.append(uci_to_id(u))
        b.push(move)
    return toks


def _print_table(tokens: List[int]) -> None:
    print(f"{'i':>4}  {'type':<6}  {'value':<14}")
    print("-" * 32)
    for i, t in enumerate(tokens):
        if is_state_token(t):
            local = t - vocab_size()
            print(f"{i:>4}  {'STATE':<6}  {local:<14} (global {t})")
        else:
            print(f"{i:>4}  {'MOVE':<6}  {id_to_uci(t):<14} (id {t})")


def cmd_info(_):
    print("=== Tokenizer info ===")
    print(f"Move vocab size   : {vocab_size()}")
    print(f"State token count : {NUM_STATE_TOKENS}")
    print(f"Total vocab size  : {vocab_total_size()} (moves + state)")
    print("\nLayout:")
    print("  Sequence layout is: [STATE, MOVE, STATE, MOVE, ...]")
    print("  STATE tokens occupy ids [vocab_size() .. vocab_size()+NUM_STATE_TOKENS-1].")
    print("  MOVE tokens map 1:1 from UCI strings.")


def cmd_encode_seq(args):
    tokens = _encode_from_seq(args.seq)
    if args.json:
        print(json.dumps({"tokens": tokens, "moves": decode_moves_only(tokens)}))
    elif args.print_table:
        _print_table(tokens)
    else:
        print("TOKENS:", tokens)
        print("MOVES :", decode_moves_only(tokens))


def cmd_decode_moves(args):
    ids = [int(x) for x in args.ids.replace(",", " ").split()]
    ucis = decode_moves_only(ids)
    if args.json:
        print(json.dumps({"decoded_ucis": ucis}))
    else:
        print("UCIs:", ucis if ucis else "(no MOVE tokens among provided ids)")


def cmd_show_state(args):
    b = chess.Board(args.fen)
    st = _statize(b)
    idx = state_index(st["stm_white"], st["castles"], st["ep_file"])
    tok = encode_state(idx)
    payload = {
        "fen": b.fen(),
        "stm_white": st["stm_white"],
        "castles": st["castles"],
        "ep_file": st["ep_file"],
        "state_index_local": idx,
        "state_token_global": tok,
        "move_vocab_size": vocab_size(),
        "total_vocab_size": vocab_total_size(),
    }
    if args.json:
        print(json.dumps(payload))
    else:
        print("FEN                  :", payload["fen"])
        print("Side to move         :", "White" if payload["stm_white"] else "Black")
        print("Castling rights      :", payload["castles"])
        print("EP file              :", payload["ep_file"])
        print("State index (local)  :", payload["state_index_local"])
        print("State token (global) :", payload["state_token_global"])
        print("Move vocab size      :", payload["move_vocab_size"])
        print("Total vocab size     :", payload["total_vocab_size"])


def main():
    ap = argparse.ArgumentParser(prog="chess-tokenizer", description="Tokenizer CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="Show tokenizer/vocab sizes and layout")

    sp_seq = sub.add_parser("encode-seq", help="Encode a space-separated UCI sequence from startpos")
    sp_seq.add_argument("--seq", required=True)
    sp_seq.add_argument("--print-table", action="store_true")
    sp_seq.add_argument("--json", action="store_true")

    sp_dec = sub.add_parser("decode-moves", help="Decode MOVE token ids back to UCI")
    sp_dec.add_argument("--ids", required=True)
    sp_dec.add_argument("--json", action="store_true")

    sp_state = sub.add_parser("show-state", help="Show the STATE token for a FEN")
    sp_state.add_argument("--fen", required=True)
    sp_state.add_argument("--json", action="store_true")

    args = ap.parse_args()
    if args.cmd == "info":
        cmd_info(args)
    elif args.cmd == "encode-seq":
        cmd_encode_seq(args)
    elif args.cmd == "decode-moves":
        cmd_decode_moves(args)
    elif args.cmd == "show-state":
        cmd_show_state(args)


if __name__ == "__main__":
    main()
