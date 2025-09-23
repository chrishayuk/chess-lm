# scripts/hello_tokenizer.py
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("out/vocab")

tokens = ["<STATE:W:KQkq:->", "e2e4", "g1f3"]

ids = tok.convert_tokens_to_ids(tokens)
print("Tokens :", tokens)
print("IDs    :", ids)
print("Back   :", tok.convert_ids_to_tokens(ids))