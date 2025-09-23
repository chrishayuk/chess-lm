# chess_lm/tokenizer/__init__.py
from .moves import (
    build_uci_catalog, 
    UCI_CATALOG, 
    UCI2ID, 
    ID2UCI, 
    uci_to_id, 
    id_to_uci, 
    vocab_size
)
from .state_tokens import state_index, NUM_STATE_TOKENS
from .encoding import (
    encode_state,
    is_state_token,
    encode_game,
    vocab_total_size,
    decode_moves_only
)
from .masks import legality_mask_batch

__all__ = [
    # moves
    'build_uci_catalog',
    'UCI_CATALOG',
    'UCI2ID',
    'ID2UCI',
    'uci_to_id',
    'id_to_uci',
    'vocab_size',
    # state tokens
    'state_index',
    'NUM_STATE_TOKENS',
    # encoding
    'encode_state',
    'is_state_token',
    'encode_game',
    'vocab_total_size',
    'decode_moves_only',
    # masks
    'legality_mask_batch'
]