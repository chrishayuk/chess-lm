import argparse
import json
import chess.pgn

def statize(board: chess.Board):
    ep = board.ep_square
    ep_file = chess.square_file(ep) if ep is not None else None
    ep_file_char = "abcdefgh"[ep_file] if ep_file is not None else None
    castles = board.castling_xfen()
    return {"stm_white": board.turn, "castles": castles if castles != "" else "-", "ep_file": ep_file_char}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.pgn, "r") as f, open(args.out, "w") as w:
        while True:
            game = chess.pgn.read_game(f)
            if game is None: break
            board = game.board()
            states, moves = [], []
            for mv in game.mainline_moves():
                states.append(statize(board))
                moves.append(mv.uci())
                board.push(mv)
            if len(moves) >= 8:
                w.write(json.dumps({"states":states, "moves":moves})+"\n")
