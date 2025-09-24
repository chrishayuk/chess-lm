# scripts/encode_smoke.py
from chess_lm.tokenizer.encoding import encode_game, decode_moves_only
from chess_lm.tokenizer.moves import id_to_uci

rec = {
    "moves": ["e2e4", "e7e5", "g1f3", "b8c6"],
    # no "states": tokenizer adds them internally
}

ids = encode_game(rec)
print("IDs:", ids)
print("Moves decoded:", decode_moves_only(ids))
print("UCIs direct :", [id_to_uci(i) for i in ids if i < 4096])  # only moves
