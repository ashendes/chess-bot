import csv
import chess
import chess.engine
import random


def generate_moves_up_to_depth(board, max_depth, engine, top_moves=5):
    """Generate a sequence of moves up to a specified depth."""
    for _ in range(max_depth):
        if board.is_game_over():  # Stop if the game is over
            break
        legal_moves = list(board.legal_moves)
        if len(legal_moves) == 0:  # No legal moves
            break
        move = random.choice(legal_moves)
        board.push(move)
        yield board.copy()


def evaluate_position(board, engine):
    """Evaluate a chess position using the provided chess engine."""
    result = engine.analyse(board, chess.engine.Limit(time=1))
    return result["score"].relative.score()


def main():
    engine_path = "/opt/homebrew/Cellar/stockfish/17/bin/stockfish"  # Update this path
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)

    try:
        with open("chess_sequential_evaluations.csv", "w", newline="") as csvfile:
            fieldnames = ["itr", "move_num", "fen", "eval"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(3000):  # Generate 10,000 games
                random.seed(i)
                board = chess.Board()
                max_moves = 4  # Random game depth

                for move_num, position in enumerate(generate_moves_up_to_depth(board, max_moves)):
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
