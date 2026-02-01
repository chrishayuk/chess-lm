from chess_lm.tokenizer import vocab_size, vocab_total_size
from chess_lm.tokenizer.state_tokens import NUM_STATE_TOKENS


def main():
    print(f"chess-lm vocabulary: {vocab_size()} moves + {NUM_STATE_TOKENS} states = {vocab_total_size()} total tokens")


if __name__ == "__main__":
    main()
