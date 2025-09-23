#!/usr/bin/env python3
"""
Export the full tokenizer vocabulary (moves + state) in several formats.

Outputs (always):
  - full_vocab.tsv      # id<TAB>token
  - moves.json          # list of move tokens (UCI strings)
  - states.json         # list of state tokens (readable strings)

Optional (--hf):
  - vocab.json              # token -> id
  - tokenizer_config.json   # minimal config for HF
  - special_tokens_map.json # if --add-specials used
  - tokenizer.json          # if 'tokenizers' is installed (WordLevel)

Examples:
  python scripts/export_vocab.py --out-dir out/vocab
  python scripts/export_vocab.py --out-dir out/vocab --hf
  python scripts/export_vocab.py --out-dir out/vocab --hf --add-specials
"""

from __future__ import annotations
import argparse
import json
import os
from typing import Dict, List, Tuple, Optional

# Your existing tokenizer API
from chess_lm.tokenizer import (
    vocab_size,          # move vocab size
    NUM_STATE_TOKENS,    # 288
    encode_state,
    state_index,
    id_to_uci,           # move id -> UCI
)

FILES = "abcdefgh"

# Stable enumeration orders for state space
CASTLE_ORDER = [
    "-", "K", "Q", "KQ", "k", "q", "kq",
    "Kk", "Kq", "Qk", "Qq", "KQk", "KQq", "Kkq", "Qkq", "KQkq"
]
EP_ORDER = [None] + list(FILES)   # None means "-"
STM_ORDER = [True, False]         # True=White, False=Black


def state_token_string(stm_white: bool, castles: str, ep_file: Optional[str]) -> str:
    side = "W" if stm_white else "B"
    ep = "-" if ep_file is None else ep_file
    cr = castles if castles else "-"
    return f"<STATE:{side}:{cr}:{ep}>"


def iter_states():
    for stm in STM_ORDER:
        for cr in CASTLE_ORDER:
            for ep in EP_ORDER:
                yield (stm, cr, ep)


def build_full_vocab(add_specials: bool) -> Tuple[List[str], Dict[str, int]]:
    """
    Returns:
      tokens: list of token strings in id order (moves + states [+ specials])
      tok2id: map token -> id
    """
    mv = vocab_size()
    tokens: List[Optional[str]] = [None] * (mv + NUM_STATE_TOKENS)

    # 1) Moves (0..mv-1)
    for i in range(mv):
        tokens[i] = id_to_uci(i)

    # 2) States (mv..)
    for stm, cr, ep in iter_states():
        idx = state_index(stm, cr, ep)
        gid = encode_state(idx)
        tokens[gid] = state_token_string(stm, cr, ep)

    # 3) Optional specials appended at the end
    specials: List[str] = []
    if add_specials:
        specials = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        tokens.extend(specials)

    # Build map
    assert all(t is not None for t in tokens), "Hole in tokens list."
    tok2id: Dict[str, int] = {t: i for i, t in enumerate(tokens)}  # type: ignore[arg-type]
    assert len(tok2id) == len(tokens), "Duplicate token strings found."

    return tokens, tok2id


def write_plain(out_dir: str, tokens: List[str]) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # TSV: id <TAB> token
    with open(os.path.join(out_dir, "full_vocab.tsv"), "w", encoding="utf-8") as f:
        for i, tok in enumerate(tokens):
            f.write(f"{i}\t{tok}\n")

    # moves.json
    mv = vocab_size()
    with open(os.path.join(out_dir, "moves.json"), "w", encoding="utf-8") as f:
        json.dump([tokens[i] for i in range(mv)], f, ensure_ascii=False, indent=2)

    # states.json
    with open(os.path.join(out_dir, "states.json"), "w", encoding="utf-8") as f:
        json.dump(tokens[mv:mv + NUM_STATE_TOKENS], f, ensure_ascii=False, indent=2)

    print(f"Plain dumps written to {out_dir}/ (full_vocab.tsv, moves.json, states.json)")


def write_hf(out_dir: str, tokens: List[str], tok2id: Dict[str, int], add_specials: bool) -> None:
    """
    Write minimal Hugging Face files:
      - vocab.json (token -> id)
      - tokenizer_config.json
      - special_tokens_map.json (if specials)
      - tokenizer.json (WordLevel) if 'tokenizers' is available
    """
    os.makedirs(out_dir, exist_ok=True)

    # vocab.json
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(tok2id, f, ensure_ascii=False, indent=2)

    # specials (optional)
    specials_map: Dict[str, str] = {}
    if add_specials:
        specials_map = {
            "pad_token": "<PAD>",
            "unk_token": "<UNK>",
            "bos_token": "<BOS>",
            "eos_token": "<EOS>",
        }
        with open(os.path.join(out_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
            json.dump(specials_map, f, ensure_ascii=False, indent=2)

    # tokenizer_config.json
    config = {
        "model_max_length": 4096,
        "do_lower_case": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
    }
    if specials_map:
        config.update(specials_map)
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # Try to produce tokenizer.json via tokenizers (WordLevel)
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.processors import TemplateProcessing

        unk_tok = specials_map.get("unk_token") if specials_map else None
        if unk_tok and unk_tok not in tok2id:
            unk_tok = None  # safety

        tk = Tokenizer(WordLevel(vocab=tok2id, unk_token=unk_tok))
        tk.pre_tokenizer = Whitespace()

        bos = specials_map.get("bos_token") if specials_map else None
        eos = specials_map.get("eos_token") if specials_map else None

        # Correct single/pair templates must always include $A (and $B for pairs)
        if bos and eos:
            single_tpl = f"{bos} $A {eos}"
            pair_tpl   = f"{bos} $A {eos} {bos} $B {eos}"
            special_list = [(bos, tok2id[bos]), (eos, tok2id[eos])]
        elif bos and not eos:
            single_tpl = f"{bos} $A"
            pair_tpl   = f"{bos} $A {bos} $B"  # must include $B
            special_list = [(bos, tok2id[bos])]
        elif eos and not bos:
            single_tpl = f"$A {eos}"
            pair_tpl   = f"$A {eos} $B {eos}"  # must include $B
            special_list = [(eos, tok2id[eos])]
        else:
            single_tpl = "$A"
            pair_tpl   = "$A $B"
            special_list = []

        tk.post_processor = TemplateProcessing(
            single=single_tpl,
            pair=pair_tpl,
            special_tokens=special_list,
        )

        tk.save(os.path.join(out_dir, "tokenizer.json"))
        print(f"Hugging Face files written to {out_dir}/ (incl. tokenizer.json)")
    except Exception as e:
        print(f"[warn] Could not write tokenizer.json: {e}")
        print("       You still have vocab.json + configs; PreTrainedTokenizerFast can load from those.")


def main():
    ap = argparse.ArgumentParser(description="Export tokenizer vocabulary (moves + state).")
    ap.add_argument("--out-dir", required=True, help="Output directory for exported files.")
    ap.add_argument("--hf", action="store_true", help="Also write Hugging Face-compatible files.")
    ap.add_argument("--add-specials", action="store_true",
                    help="Append <PAD>, <UNK>, <BOS>, <EOS> at the end of the vocab.")
    args = ap.parse_args()

    tokens, tok2id = build_full_vocab(add_specials=args.add_specials)
    write_plain(args.out_dir, tokens)

    if args.hf:
        write_hf(args.out_dir, tokens, tok2id, add_specials=args.add_specials)


if __name__ == "__main__":
    main()
