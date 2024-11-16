import csv, chess, chess.engine, random

def generate_random_position():
    board = chess.Board()
    for _ in range(random.randint(10, 50)):
        legal_moves = list(board.legal_moves)
        if len(legal_moves) == 0:
            break
        move = random.choice(legal_moves)
        board.push(move)
    return board

def evaluate_position(board, engine):
    result = engine.analyse(board, chess.engine.Limit(time=1))
    return result['score'].relative.score()

def main():
    engine = chess.engine.SimpleEngine.popen_uci("/path/to/stockfish")  # Change this path
    try:
        with open('chess_evaluations.csv', 'w', newline='') as csvfile:
            fieldnames = ['itr', 'fen', 'eval']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(100):
                random.seed(i)
                board = generate_random_position()
                evaluation = evaluate_position(board, engine)
                if evaluation != None:
                    writer.writerow({'itr': i, 'fen': board.fen(), 'eval': evaluation})
    finally:
        engine.quit()

if __name__ == "__main__":
    main()
