import csv
import chess
import chess.engine
import random


def generate_moves_up_to_depth(board, max_depth, engine, top_moves=5):
    """Generate a sequence of moves up to a specified depth."""
    for _ in range(max_depth):
        if board.is_game_over():
            break
        legal_moves = list(board.legal_moves)
        if len(legal_moves) == 0:
            break

        move_scores = []
        for move in legal_moves[:top_moves]:
            board.push(move)
            eval_score = evaluate_position(board, engine)
            move_scores.append((move, eval_score))
            board.pop()

        move_scores.sort(key=lambda x: x[1], reverse=board.turn)

        selected_move = random.choice(move_scores[:top_moves])[0] if move_scores else random.choice(legal_moves)
        board.push(selected_move)
        yield board.copy()


def evaluate_position(board, engine):
    """Evaluate a chess position using the provided chess engine."""
    result = engine.analyse(board, chess.engine.Limit(time=0.2))
    return result["score"].relative.score()


def main():
    """Generate evaluated FEN positions and write them to CSV."""
    engine_path = os.environ.get("STOCKFISH_PATH")
    if not engine_path:
        raise ValueError("STOCKFISH_PATH is not set. Provide the engine path via env var.")
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)

    try:
        with open("chess_evaluations.csv", "w", newline="") as csvfile:
            fieldnames = ["itr", "move_num", "fen", "eval"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(3000):
                random.seed(i)
                board = chess.Board()
                max_moves = 4

                for move_num, position in enumerate(generate_moves_up_to_depth(board, max_moves, engine)):
                    evaluation = evaluate_position(position, engine)
                    if evaluation is not None:
                        writer.writerow({
                            "itr": i,
                            "move_num": move_num,
                            "fen": position.fen(),
                            "eval": evaluation
                        })
    finally:
        engine.quit()


if __name__ == "__main__":
    main()
