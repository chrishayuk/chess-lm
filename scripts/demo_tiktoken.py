#!/usr/bin/env python3
"""
Quick tiktoken demo: show how text is tokenized into IDs,
then compare with chess UCI moves being tokenized.
"""

import tiktoken

# Part 1: normal language tokenization
enc = tiktoken.get_encoding("cl100k_base")

text = "Hello world, this is chess!"
tokens = enc.encode(text)
decoded = enc.decode(tokens)

print("=== Language Example ===")
print("Text   :", text)
print("Tokens :", tokens)
print("Decoded:", decoded)
print()

# Part 2: chess move tokenization (mocked)
# Pretend we have a fixed vocab mapping UCIs -> IDs
UCI2ID = {"e2e4": 1234, "e7e5": 987, "g1f3": 2718}
ID2UCI = {v: k for k, v in UCI2ID.items()}

moves = ["e2e4", "e7e5", "g1f3"]
move_ids = [UCI2ID[m] for m in moves]

print("=== Chess Example ===")
print("UCI moves :", moves)
print("Token IDs :", move_ids)
print("Decoded   :", [ID2UCI[i] for i in move_ids])
