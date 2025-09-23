#!/usr/bin/env python3
"""
Example: using the exported chess tokenizer with Hugging Face.

Shows:
  - Loading tokenizer from out/vocab/
  - Converting chess state/move tokens to IDs and back
  - Creating a padded batch (input_ids + attention_mask)
"""

from transformers import AutoTokenizer
import torch

def main():
    # 1. Load tokenizer
    tok = AutoTokenizer.from_pretrained("out/vocab")
    print("PAD id:", tok.pad_token_id)
    print("UNK id:", tok.unk_token_id)

    # 2. Example chess sequences (already tokenized!)
    game1 = ["<STATE:W:KQkq:->", "e2e4",
             "<STATE:B:KQkq:e>", "e7e5",
             "<STATE:W:KQkq:e>", "g1f3"]

    game2 = ["<STATE:W:KQkq:->", "d2d4",
             "<STATE:B:KQkq:d>", "d7d5"]

    batch_tokens = [game1, game2]

    # 3. Convert tokens -> IDs directly
    batch_ids = [tok.convert_tokens_to_ids(seq) for seq in batch_tokens]

    # Debug: list any UNKs
    for i, (seq, ids) in enumerate(zip(batch_tokens, batch_ids)):
        unk_id = tok.unk_token_id
        bad = [(t, tid) for t, tid in zip(seq, ids) if tid == unk_id]
        if bad:
            print(f"[row {i}] UNK tokens:", bad)

    # 4. Pad to same length
    maxlen = max(len(ids) for ids in batch_ids)
    PAD = tok.pad_token_id
    padded = [ids + [PAD] * (maxlen - len(ids)) for ids in batch_ids]
    attn   = [[1] * len(ids) + [0] * (maxlen - len(ids)) for ids in batch_ids]

    input_ids = torch.tensor(padded, dtype=torch.long)
    attention = torch.tensor(attn, dtype=torch.bool)

    print("\n=== Batch ===")
    print("input_ids:\n", input_ids)
    print("attention_mask:\n", attention)

    # 5. Convert back to tokens (strip pads)
    for i, row in enumerate(input_ids.tolist()):
        tokens = tok.convert_ids_to_tokens(row)
        no_pads = [t for t in tokens if t != tok.pad_token]
        print(f"\nRow {i} back to tokens:", no_pads)

if __name__ == "__main__":
    main()
