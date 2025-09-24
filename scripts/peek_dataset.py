# scripts/peek_dataset.py
from transformers import AutoTokenizer
from chess_lm.data.chess_sequence_dataset import ChessSequenceDataset

tok = AutoTokenizer.from_pretrained("out/vocab")
ds = ChessSequenceDataset(
    path="data/test_games.jsonl",
    max_len=128,
    stride=64,
    seed=1337,
    start_on_state=True,
    drop_trailing_state=True,
    lazy_load=False,   # set True if the file is huge
    validate_tokens=True,
    pad_short_sequences=False,
    return_info=False,
    verbose=True,
)

print("windows:", len(ds))
ids = ds[0]
print("len(ids):", len(ids))
print("tokens  :", tok.convert_ids_to_tokens(ids)[:40])
