
## chess_lm/tokenizer/moves.py
# maps every plausible move (uci) to a stable ID and back
from __future__ import annotations
import itertools

# We include all (from,to) pairs on 8x8 plus promotions for pawns.
# Filtering to legal happens at runtime via masks; the catalog just enumerates.
FILES = "abcdefgh"
RANKS = "12345678"
SQUARES = [f + r for r in RANKS for f in FILES]
PROMOS = ["q", "r", "b", "n"]

# Build a superset catalog like UCI: e2e4, a7a8q, etc.
def build_uci_catalog() -> list[str]:
    base = []
    for s in SQUARES:
        for t in SQUARES:
            if s != t:
                base.append(s + t)  # quiet/captures, castles included as e1g1/e1c1 etc.
    # Add underpromotions (both white & black directions will be masked by legality)
    promos = []
    for f in FILES:
        promos += [f+"7"+f+"8"+p for p in PROMOS]  # white
        promos += [f+"2"+f+"1"+p for p in PROMOS]  # black
        # Include diagonal promo captures in catalog; legality will filter later
        for df in (-1, +1):
            idx = FILES.index(f)
            if 0 <= idx+df < 8:
                tf = FILES[idx+df]
                promos += [f+"7"+tf+"8"+p for p in PROMOS]
                promos += [f+"2"+tf+"1"+p for p in PROMOS]
    uci = sorted(set(base + promos))
    return uci

UCI_CATALOG = build_uci_catalog()
UCI2ID = {u:i for i,u in enumerate(UCI_CATALOG)}
ID2UCI = {i:u for u,i in UCI2ID.items()}

def uci_to_id(uci: str) -> int:
    return UCI2ID[uci]

def id_to_uci(i: int) -> str:
    return ID2UCI[i]

def vocab_size() -> int:
    return len(UCI_CATALOG)
