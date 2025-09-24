# chess_lm/data/collate_causal.py
from dataclasses import dataclass
from typing import List, Optional

import torch

# chess_lm
from chess_lm.tokenizer.encoding import is_state_token


@dataclass
class CollateCausal:
    """
    Causal LM collator for chess token sequences.

    Always returns:
      - input_ids:      [B, T] long
      - attention_mask: [B, T] bool
      - labels:         [B, T] long (pads/masked positions set to ignore_index)
      - loss_mask:      [B, T] bool (True where labels are trained)
      - boundary_mask:  [B, T] bool (optional) True where CURRENT token is a STATE

    Masking modes:
      - "none"       : train on all non-pad tokens (default; HF-style)
      - "only_moves" : train only where we're predicting a MOVE
                       i.e., positions j where previous token (j-1) is STATE
      - "mask_sim"   : zero loss inside <SIM_BEGIN> ... <SIM_END> spans
                       (requires sim_begin_id & sim_end_id; if absent, silently skipped)
    """

    pad_id: int
    ignore_index: int = -100
    make_boundary_mask: bool = True

    # Optional test-time/aux features
    sim_begin_id: Optional[int] = None
    sim_end_id: Optional[int] = None
    mask_mode: str = "none"  # "none" | "only_moves" | "mask_sim"
    max_seq_len: Optional[int] = None  # truncate right if set
    device: Optional[torch.device] = None

    def __call__(self, batch: List[List[int]]):
        # Truncate if needed
        if self.max_seq_len is not None:
            batch = [seq[: self.max_seq_len] for seq in batch]

        B = len(batch)
        T = max((len(x) for x in batch), default=0)

        input_ids = torch.full((B, T), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, T), dtype=torch.bool)
        boundary_mask = torch.zeros((B, T), dtype=torch.bool) if self.make_boundary_mask else None

        # Fill inputs/masks
        for b, seq in enumerate(batch):
            L = len(seq)
            if L == 0:
                continue
            input_ids[b, :L] = torch.as_tensor(seq, dtype=torch.long)
            attention_mask[b, :L] = True
            if boundary_mask is not None:
                boundary_mask[b, :L] = torch.as_tensor([is_state_token(t) for t in seq], dtype=torch.bool)

        # Base predict mask (train wherever we have a real token)
        predict_mask = attention_mask.clone()

        # --- Masking strategies ---
        if self.mask_mode == "only_moves":
            # Keep loss only where we predict a MOVE:
            # positions j (>=1) whose *previous* token is STATE
            only_moves = torch.zeros_like(predict_mask)
            for b, seq in enumerate(batch):
                for j in range(1, len(seq)):
                    if is_state_token(seq[j - 1]):
                        only_moves[b, j] = True
            predict_mask &= only_moves

        elif self.mask_mode == "mask_sim":
            # Zero loss INSIDE <SIM_BEGIN> ... <SIM_END>; skip if ids not provided
            if self.sim_begin_id is not None and self.sim_end_id is not None:
                for b, seq in enumerate(batch):
                    inside = False
                    for j, tid in enumerate(seq):
                        if tid == self.sim_begin_id:
                            inside = True
                        elif tid == self.sim_end_id:
                            inside = False
                        if inside:
                            predict_mask[b, j] = False
            # else: silently no-op (safe for vocabs without thinking tokens)

        # Labels: copy ids, then ignore where we don't train
        labels = input_ids.clone()
        labels[~predict_mask] = self.ignore_index

        # Optionally move tensors to device
        if self.device is not None:
            input_ids = input_ids.to(self.device, non_blocking=True)
            attention_mask = attention_mask.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            predict_mask = predict_mask.to(self.device, non_blocking=True)
            if boundary_mask is not None:
                boundary_mask = boundary_mask.to(self.device, non_blocking=True)

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "loss_mask": predict_mask,
        }
        if boundary_mask is not None:
            out["boundary_mask"] = boundary_mask
        return out
