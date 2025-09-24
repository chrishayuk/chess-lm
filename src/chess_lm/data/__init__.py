# chess_lm/data/__init__.py
"""
Data loading utilities for chess_lm.

Exports:
    - ChessSequenceDataset: JSONL → overlapping token windows
    - DatasetStats: summary stats dataclass
"""

from .chess_sequence_dataset import ChessSequenceDataset, DatasetStats

__all__ = [
    "ChessSequenceDataset",
    "DatasetStats",
]
