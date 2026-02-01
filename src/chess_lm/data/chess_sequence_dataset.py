# chess_lm/data/chess_sequence_dataset.py
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import chess  # NEW
import numpy as np
from torch.utils.data import Dataset

from chess_lm.tokenizer import (
    encode_game,
    is_state_token,
    vocab_total_size,
)
from chess_lm.tokenizer.moves import id_to_uci  # NEW

logger = logging.getLogger(__name__)


@dataclass
class DatasetStats:
    """Statistics about the dataset."""

    total_games: int
    total_windows: int
    total_tokens: int
    avg_game_length: float
    min_game_length: int
    max_game_length: int
    filtered_games: int
    vocab_size: int


class ChessSequenceDataset(Dataset):
    """
    Efficient PyTorch Dataset for chess game sequences.

    Loads chess games from JSONL format and creates overlapping sequence windows
    suitable for training sequence models.

    Features:
      - Eager and lazy modes
      - Configurable windowing with overlap
      - Optional state-aligned starts
      - Optional drop of trailing lone STATE tokens
      - Cache support (eager mode)
      - Reproducible shuffling via seed
      - Optional per-item metadata via return_info
      - Optional per-window initial FEN export for legality masks
    """

    def __init__(
        self,
        path: Union[str, Path],
        max_len: int = 512,
        stride: Optional[int] = None,
        seed: Optional[int] = 1337,
        start_on_state: bool = True,
        min_game_length: int = 4,
        lazy_load: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
        validate_tokens: bool = True,
        pad_short_sequences: bool = False,
        pad_token_id: Optional[int] = None,
        drop_trailing_state: bool = True,
        max_games: Optional[int] = None,
        return_info: bool = False,
        verbose: bool = True,
        emit_initial_fen: bool = False,  # NEW: store FEN for each window start
    ):
        """
        Args:
            path: JSONL file of games (one JSON object per line).
            max_len: Window length in tokens.
            stride: Hop between window starts (default: max_len // 2).
            seed: Random seed for reproducible shuffling/stat sampling.
            start_on_state: Align window starts to a STATE token if possible.
            min_game_length: Drop games shorter than this many tokens.
            lazy_load: If True, don't materialize windows; store indices/offsets.
            cache_dir: If set (and eager mode), save/load encoded windows to/from cache.
            validate_tokens: Check token ids are within [0, vocab_size).
            pad_short_sequences: If True, right-pad windows shorter than max_len.
            pad_token_id: Required if pad_short_sequences=True. Must not collide with real ids.
            drop_trailing_state: If a window ends on a lone STATE token, drop it so next target is a MOVE.
            max_games: Limit number of games read (useful for debug).
            return_info: If True, __getitem__ returns dict with metadata; else returns List[int].
            verbose: Log dataset statistics.
            emit_initial_fen: If True, include "initial_fen" (FEN before window start) in window info.
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")

        self.max_len = int(max_len)
        if self.max_len <= 0:
            raise ValueError("max_len must be positive")

        self.stride = int(stride if stride is not None else max_len // 2)
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.stride > self.max_len:
            raise ValueError(f"stride ({self.stride}) cannot exceed max_len ({self.max_len})")

        self.min_game_length = int(min_game_length)
        self.start_on_state = bool(start_on_state)
        self.lazy_load = bool(lazy_load)
        self.validate_tokens = bool(validate_tokens)
        self.pad_short_sequences = bool(pad_short_sequences)
        self.pad_token_id = pad_token_id
        self.drop_trailing_state = bool(drop_trailing_state)
        self.max_games = max_games
        self.return_info = bool(return_info)
        self.verbose = bool(verbose)
        self.emit_initial_fen = bool(emit_initial_fen)  # NEW

        # Vocab & seeding
        self.vocab_size = vocab_total_size()
        self.py_rng = random.Random(seed) if seed is not None else random.Random()
        if seed is not None:
            np.random.seed(seed)

        # Padding safety
        if self.pad_short_sequences:
            if self.pad_token_id is None:
                raise ValueError("pad_short_sequences=True requires pad_token_id (non-colliding).")
            if not (0 <= self.pad_token_id < self.vocab_size):
                raise ValueError(f"pad_token_id {self.pad_token_id} out of bounds [0,{self.vocab_size}).")

        # Cache
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Build
        if self.lazy_load:
            self._init_lazy()
        else:
            self._init_eager()

        # Stats
        self.stats = self._compute_stats()
        if self.verbose:
            self._log_stats()

    # ------------------------- Eager init & caching -------------------------

    def _cache_key(self) -> Optional[Path]:
        if not self.cache_dir:
            return None
        key = (
            f"{self.path.stem}"
            f"_ml{self.max_len}"
            f"_st{self.stride}"
            f"_sos{int(self.start_on_state)}"
            f"_mgl{self.min_game_length}"
            f"_dts{int(self.drop_trailing_state)}"
            f"_pad{int(self.pad_short_sequences)}"
            f"_eif{int(self.emit_initial_fen)}"  # NEW: emit_initial_fen affects window_info schema
        )
        return self.cache_dir / f"{key}.npz"

    def _init_eager(self):
        cache_path = self._cache_key()
        if cache_path and cache_path.exists():
            try:
                if self.verbose:
                    logger.info(f"Loading dataset cache: {cache_path}")
                data = np.load(cache_path, allow_pickle=True)
                self.items = [list(arr) for arr in data["items"]]
                self.window_info = data["window_info"].tolist()
                self._filtered_games = int(data.get("filtered_games", 0))
                return
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_path}: {e}")

        self.items: List[List[int]] = []
        self.window_info: List[Dict[str, Any]] = []
        self._filtered_games = 0
        games_processed = 0

        with open(self.path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                if self.max_games is not None and games_processed >= self.max_games:
                    break
                try:
                    game = json.loads(line)
                    toks = encode_game(game)
                except Exception as e:
                    logger.warning(f"[line {line_num}] failed to parse/encode game: {e}")
                    continue

                if len(toks) < self.min_game_length:
                    self._filtered_games += 1
                    continue

                if self.validate_tokens:
                    self._validate_tokens(toks, line_num)

                windows = self._create_windows(toks, game_id=games_processed)
                self.items.extend([w["tokens"] for w in windows])
                self.window_info.extend(windows)
                games_processed += 1

        # Shuffle windows deterministically
        if self.items:
            idx = list(range(len(self.items)))
            self.py_rng.shuffle(idx)
            self.items = [self.items[i] for i in idx]
            self.window_info = [self.window_info[i] for i in idx]

        # Save cache
        if cache_path and self.items:
            try:
                np.savez_compressed(
                    cache_path,
                    items=np.array([np.array(it, dtype=np.int32) for it in self.items], dtype=object),
                    window_info=np.array(self.window_info, dtype=object),
                    filtered_games=np.array(self._filtered_games, dtype=np.int64),
                )
                if self.verbose:
                    logger.info(f"Saved dataset cache → {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save cache {cache_path}: {e}")

    # --------------------------- Lazy init ----------------------------------

    def _init_lazy(self):
        self.game_offsets: List[Tuple[int, int]] = []  # (file_offset, line_len)
        self.window_indices: List[Tuple[int, int, int]] = []  # (game_idx, start_pos, end_pos)
        games_processed = 0
        offset = 0

        with open(self.path, "rb") as f:
            for line in f:
                if self.max_games is not None and games_processed >= self.max_games:
                    break
                line_len = len(line)
                try:
                    game = json.loads(line.decode("utf-8"))
                    toks = encode_game(game)
                except Exception as e:
                    logger.warning(f"[lazy] skip game at byte {offset}: {e}")
                    offset += line_len
                    continue

                if len(toks) >= self.min_game_length:
                    self.game_offsets.append((offset, line_len))
                    # compute window indices for this game
                    start_pos = 0
                    if self.start_on_state:
                        while start_pos < len(toks) and not is_state_token(toks[start_pos]):
                            start_pos += 1

                    while start_pos < len(toks):
                        end_pos = min(start_pos + self.max_len, len(toks))
                        # optionally drop a trailing lone STATE at end
                        if self.drop_trailing_state and end_pos - start_pos >= 1:
                            if is_state_token(toks[end_pos - 1]):
                                end_pos -= 1
                        if end_pos > start_pos:
                            self.window_indices.append((games_processed, start_pos, end_pos))
                        start_pos += self.stride
                        if start_pos >= len(toks):
                            break

                    games_processed += 1

                offset += line_len

        # Shuffle window order reproducibly
        self.py_rng.shuffle(self.window_indices)

    # ------------------------- Window construction --------------------------

    def _fen_before(self, tokens: List[int], idx: int) -> str:
        """
        FEN of the position just BEFORE tokens[idx].
        Starts from standard initial position and applies only *legal* MOVE tokens
        up to idx-1. Robust to occasional noisy/illegal tokens (they're skipped).
        """
        bd = chess.Board()
        j = 0
        while j < idx:
            t = tokens[j]
            if not is_state_token(t):
                try:
                    mv = chess.Move.from_uci(id_to_uci(t))
                except Exception:
                    j += 1
                    continue
                if mv in bd.legal_moves:
                    bd.push(mv)
            j += 1
        return bd.fen()

    def _create_windows(self, tokens: List[int], game_id: int) -> List[Dict[str, Any]]:
        windows: List[Dict[str, Any]] = []

        # align start
        start_pos = 0
        if self.start_on_state:
            while start_pos < len(tokens) and not is_state_token(tokens[start_pos]):
                start_pos += 1

        while start_pos < len(tokens):
            end_pos = min(start_pos + self.max_len, len(tokens))

            # optionally drop trailing lone STATE
            if self.drop_trailing_state and end_pos - start_pos >= 1:
                if is_state_token(tokens[end_pos - 1]):
                    end_pos -= 1

            if end_pos <= start_pos:  # nothing to add
                start_pos += self.stride
                continue

            window_tokens = tokens[start_pos:end_pos]

            # dataset-level padding (usually leave to collator)
            padded = False
            if self.pad_short_sequences and len(window_tokens) < self.max_len:
                assert self.pad_token_id is not None
                pad_needed = self.max_len - len(window_tokens)
                window_tokens = window_tokens + [self.pad_token_id] * pad_needed
                padded = True

            win: Dict[str, Any] = {
                "tokens": window_tokens,
                "game_id": game_id,
                "start_pos": start_pos,
                "end_pos": end_pos,
                "length": len(window_tokens),
                "padded": padded,
            }

            # NEW: attach the initial FEN of this window (position before start_pos)
            if self.emit_initial_fen:
                win["initial_fen"] = self._fen_before(tokens, start_pos)

            windows.append(win)

            start_pos += self.stride
            if start_pos >= len(tokens):
                break

        return windows

    # ------------------------------- Utils ----------------------------------

    def _validate_tokens(self, tokens: List[int], line_num: Optional[int] = None):
        for i, t in enumerate(tokens):
            if not (0 <= t < self.vocab_size):
                where = f" (line {line_num})" if line_num is not None else ""
                raise ValueError(f"Token {t} at pos {i}{where} out of bounds [0,{self.vocab_size}).")

    def _load_game(self, game_idx: int) -> List[int]:
        """Load and encode a specific game (for lazy loading)."""
        offset, length = self.game_offsets[game_idx]
        with open(self.path, "rb") as f:
            f.seek(offset)
            line = f.read(length)
        game: Dict[str, Any] = json.loads(line.decode("utf-8"))
        result: List[int] = encode_game(game)
        return result

    # ------------------------------- Stats ----------------------------------

    def _compute_stats(self) -> DatasetStats:
        if self.lazy_load:
            total_games = len(self.game_offsets)
            total_windows = len(self.window_indices)

            # unbiased estimate: sample up to 100 random games
            if total_games == 0:
                return DatasetStats(
                    total_games=0,
                    total_windows=0,
                    total_tokens=0,
                    avg_game_length=0.0,
                    min_game_length=0,
                    max_game_length=0,
                    filtered_games=0,
                    vocab_size=self.vocab_size,
                )

            sample_n = min(100, total_games)
            sample_idxs = self.py_rng.sample(range(total_games), sample_n)
            lengths = []
            for gi in sample_idxs:
                toks = self._load_game(gi)
                lengths.append(len(toks))

            avg_len = float(np.mean(lengths)) if lengths else 0.0
            min_len = int(np.min(lengths)) if lengths else 0
            max_len = int(np.max(lengths)) if lengths else 0
            est_total_tokens = int(avg_len * total_games)

            return DatasetStats(
                total_games=total_games,
                total_windows=total_windows,
                total_tokens=est_total_tokens,
                avg_game_length=avg_len,
                min_game_length=min_len,
                max_game_length=max_len,
                filtered_games=0,
                vocab_size=self.vocab_size,
            )

        # Eager mode: exact
        if not getattr(self, "window_info", None):
            return DatasetStats(
                total_games=0,
                total_windows=0,
                total_tokens=0,
                avg_game_length=0.0,
                min_game_length=0,
                max_game_length=0,
                filtered_games=0,
                vocab_size=self.vocab_size,
            )

        game_ids = set(w["game_id"] for w in self.window_info)
        # approximate game length as max end_pos seen per game
        last_pos: Dict[int, int] = {}
        for w in self.window_info:
            gid = w["game_id"]
            last_pos[gid] = max(last_pos.get(gid, 0), w["end_pos"])

        lengths = list(last_pos.values())
        return DatasetStats(
            total_games=len(game_ids),
            total_windows=len(self.items),
            total_tokens=sum(len(item) for item in self.items),
            avg_game_length=float(np.mean(lengths)) if lengths else 0.0,
            min_game_length=int(np.min(lengths)) if lengths else 0,
            max_game_length=int(np.max(lengths)) if lengths else 0,
            filtered_games=int(getattr(self, "_filtered_games", 0)),
            vocab_size=self.vocab_size,
        )

    def _log_stats(self):
        s = self.stats
        logger.info(f"Dataset: {self.path.name}")
        logger.info(f"  Total games     : {s.total_games:,}")
        logger.info(f"  Total windows   : {s.total_windows:,}")
        logger.info(f"  Total tokens    : {s.total_tokens:,}")
        logger.info(f"  Avg game length : {s.avg_game_length:.1f}")
        logger.info(f"  Game len range  : [{s.min_game_length}, {s.max_game_length}]")
        logger.info(f"  Filtered games  : {s.filtered_games:,}")
        logger.info(f"  Vocab size      : {s.vocab_size:,}")
        logger.info(
            f"  Window size     : {self.max_len}, Stride: {self.stride}, StartOnState: {self.start_on_state}, DropTrailingState: {self.drop_trailing_state}, EmitInitialFEN: {self.emit_initial_fen}"
        )

    # ---------------------------- PyTorch API --------------------------------

    def __len__(self) -> int:
        return len(self.window_indices) if self.lazy_load else len(self.items)

    def __getitem__(self, idx: int) -> Union[List[int], Dict[str, Any]]:
        if self.lazy_load:
            game_idx, start_pos, end_pos = self.window_indices[idx]
            toks = self._load_game(game_idx)
            window = toks[start_pos:end_pos]

            # Optional drop trailing state here, too (paranoia)
            if self.drop_trailing_state and window and is_state_token(window[-1]):
                window = window[:-1]

            padded = False
            if self.pad_short_sequences and len(window) < self.max_len:
                assert self.pad_token_id is not None
                pad_needed = self.max_len - len(window)
                window = window + [self.pad_token_id] * pad_needed
                padded = True

            if self.return_info:
                info: Dict[str, Any] = {
                    "tokens": window,
                    "game_id": game_idx,
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                    "length": len(window),
                    "padded": padded,
                }
                if self.emit_initial_fen:
                    info["initial_fen"] = self._fen_before(toks, start_pos)  # NEW
                return info
            return window

        # Eager
        tokens: List[int] = self.items[idx]
        if self.return_info:
            info = dict(self.window_info[idx])  # shallow copy
            info["tokens"] = tokens
            return info
        return tokens
