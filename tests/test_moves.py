import pytest
from chess_lm.tokenizer import (
    build_uci_catalog, UCI_CATALOG, UCI2ID, ID2UCI, uci_to_id, id_to_uci, vocab_size
)


def test_catalog_nonempty_and_unique():
    uci = UCI_CATALOG
    assert len(uci) > 2000, "Catalog should be a few thousand entries"
    assert len(uci) == len(set(uci)), "Catalog must be unique"


def test_roundtrip_common_moves():
    samples = ["e2e4", "e7e5", "g1f3", "b8c6", "e1g1", "e8c8"]  # includes castles by UCI
    for mv in samples:
        mid = uci_to_id(mv)
        mv2 = id_to_uci(mid)
        assert mv == mv2


def test_roundtrip_promotions():
    samples = ["a7a8q", "b7a8n", "g2g1r", "h2g1b"]
    for mv in samples:
        mid = uci_to_id(mv)
        assert id_to_uci(mid) == mv


def test_vocab_size_matches_maps():
    V = vocab_size()
    assert V == len(UCI_CATALOG) == len(UCI2ID) == len(ID2UCI)