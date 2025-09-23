#!/usr/bin/env python3
"""
Load the exported chess tokenizer (moves + state) with Hugging Face.

- If tokenizer.json exists: AutoTokenizer.from_pretrained(path)
- Else: build a tokenizers.WordLevel from vocab.json (+ specials) and wrap in
        transformers.PreTrainedTokenizerFast without triggering slow->fast conversion.
"""

from __future__ import annotations
import argparse, json, os, sys

def load_tokenizer(path: str):
    tokjson = os.path.join(path, "tokenizer.json")
    if os.path.exists(tokjson):
        from transformers import AutoTokenizer
        print("[info] loading via AutoTokenizer.from_pretrained (tokenizer.json)")
        return AutoTokenizer.from_pretrained(path)

    # Fallback: build a WordLevel tokenizer programmatically
    vocab_file = os.path.join(path, "vocab.json")
    if not os.path.exists(vocab_file):
        print(f"[error] {vocab_file} not found. Run export_vocab.py first.")
        sys.exit(1)

    try:
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
    except Exception as e:
        print("[error] This loader needs 'tokenizers' and 'transformers'.")
        print("        Install with: uv add tokenizers transformers")
        print("        (Or export tokenizer.json and use AutoTokenizer.)")
        sys.exit(1)

    with open(vocab_file, "r", encoding="utf-8") as f:
        tok2id = json.load(f)

    # Read optional specials
    specials_path = os.path.join(path, "special_tokens_map.json")
    specials = {}
    if os.path.exists(specials_path):
        with open(specials_path, "r", encoding="utf-8") as f:
            specials = json.load(f)

    unk_tok = specials.get("unk_token")
    tk = Tokenizer(WordLevel(vocab=tok2id, unk_token=unk_tok))
    tk.pre_tokenizer = Whitespace()

    print("[info] loading via PreTrainedTokenizerFast(tokenizer_object=WordLevel)")
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        **{k: v for k, v in specials.items() if k.endswith("_token")}
    )
    return fast

def main():
    ap = argparse.ArgumentParser(description="Load exported chess tokenizer (HF-compatible).")
    ap.add_argument("--path", required=True, help="Directory containing vocab.json / tokenizer.json, etc.")
    ap.add_argument("--tokens", type=str, default="e2e4 <STATE:W:KQkq:-> g1f3",
                    help="Space-separated tokens to encode/decode for a quick sanity check.")
    args = ap.parse_args()

    tok = load_tokenizer(args.path)

    pieces = args.tokens.strip().split()
    ids = tok.convert_tokens_to_ids(pieces)
    back = tok.convert_ids_to_tokens(ids)

    print("\n=== Sanity check ===")
    print("Tokens in :", pieces)
    print("IDs       :", ids)
    print("Tokens out:", back)

    # Show a few lookups if present
    for t in ["e2e4", "g1f3", "<STATE:W:KQkq:->", "<STATE:B:-:->", "<PAD>", "<UNK>"]:
        if t in tok.get_vocab():
            print(f"{t:<20} -> {tok.convert_tokens_to_ids(t)}")

    print("\n[ok] tokenizer loaded.")

if __name__ == "__main__":
    main()
