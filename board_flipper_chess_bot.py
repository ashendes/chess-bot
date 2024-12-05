#!/usr/bin/env python
import chess
import chess.svg
import sys
import pickle

MAX_DEPTH = 6
NO_TOP_MOVES = 6
board = chess.Board()

def load_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

model = load_model('chess-carnage-svr.pkl')

def evaluate_board(b: chess.Board):
    """
    Evaluate the chess board position based on material value and piece position.

    Parameters:
    - b (chess.Board): The current chess board position.

    Returns:
    - int: An integer representing the evaluation score of the board.
           Positive values favor White, and negative values favor Black.
    """
    """
    Evaluates a position using the learned model.
    """
    features = fen_to_features(b.fen())
    evaluation = model.predict([features])
    return evaluation[0]

def fen_to_features(fen):
    """
    Converts a FEN string to a feature vector suitable for machine learning.
    In this example, we use a simple feature representation where each piece on
    the board is encoded as a binary feature.
    """
    board = chess.Board(fen)
    feature_vector = []

    piece_map = board.piece_map()
    for square in chess.SQUARES:
        piece = piece_map.get(square)
        if piece:
            piece_type = piece.piece_type
            piece_color = piece.color
            # Encode piece type (1-6) and color (1 for white, 2 for black)
            feature_vector.append((piece_type, 1 if piece_color else 2))
        else:
            feature_vector.append((0, 0))
    # Flatten the feature vector
    feature_vector_flat = [val for sublist in feature_vector for val in sublist]
    # Include whose turn it is (0 for white's turn, 1 for black's turn)
    turn_feature = 0 if board.turn == chess.WHITE else 1
    feature_vector_flat.append(turn_feature)
    return feature_vector_flat

def get_top_moves(b: chess.Board, is_maximizing, no_top_moves):
    """
    Get the top moves for a given board position based on evaluation scores.

    Parameters:
    - b (chess.Board): The current chess board position.
    - is_maximizing (bool): True if it's White's turn (maximizing player), False if it's Black's turn (minimizing player).
    - no_top_moves (int): The maximum number of moves to consider based on evaluation scores.

    Returns:
    - list: A list of moves (chess.Move) representing the top moves in order of their evaluation score.
    """
    move_scores = []
    for move in b.legal_moves:
        b.push(move)
        score = evaluate_board(b)
        move_scores.append((score, move))
        b.pop()
    move_scores.sort(reverse=is_maximizing, key=lambda x: x[0])
    top_moves = move_scores[:no_top_moves]
    return [move for score, move in top_moves]

def minimax_alphabeta(board: chess.Board, depth, alpha, beta, is_maximizing, no_top_moves,
                            move=None, prune=True):
    """
    Perform a minimax search with alpha-beta pruning and returns the best move based on the evaluation scores.

    Parameters:
    - board (chess.Board): The current chess board position.
    - depth (int): The remaining depth of the search.
    - alpha (float): The alpha value for alpha-beta pruning.
    - beta (float): The beta value for alpha-beta pruning.
    - is_maximizing (bool): True if maximizing player's turn (White), False if minimizing player's turn (Black).
    - no_top_moves (int): Number of top moves to consider at each level.
    - move (chess.Move): The move leading to this board position.
    - prune (bool): If True, use alpha-beta pruning; otherwise, perform a regular minimax search.

    Returns:
    - int: The selected evaluation score for the board position.
    """

    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    top_moves = get_top_moves(board, is_maximizing, no_top_moves)

    best_eval = -float("inf") if is_maximizing else float("inf")
    for move in top_moves:
        board.push(move)
        evaluation = minimax_alphabeta(board, depth - 1, alpha, beta, not is_maximizing, no_top_moves, move)
        board.pop()
        if is_maximizing:
            best_eval = max(best_eval, evaluation)
            alpha = max(alpha, best_eval) if prune else alpha
        else:
            best_eval = min(best_eval, evaluation)
            beta = min(beta, best_eval) if prune else beta
        if alpha >= beta:
            break
    return best_eval

def make_them_flip_the_board(b: chess.Board, depth, no_top_moves, prune=True):
    """
    Generate the minimax game tree using alpha-beta pruning and return the best move.

    Parameters:
    - b (chess.Board): The current chess board position.
    - depth (int): The depth of the minimax search.
    - no_top_moves (int): Number of top moves to consider at each level.
    - prune (bool): If True, use alpha-beta pruning.

    Returns:
    - move (chess.Move): The best move determined by the minimax algorithm.
    - G (nx.DiGraph): The directed graph representing the minimax game tree.
    """

    is_maximizing = b.turn
    best_move = None
    best_value = -float("inf") if is_maximizing else float('inf')
    alpha = -float("inf")
    beta = float("inf")

    top_moves = get_top_moves(b, is_maximizing, no_top_moves)

    for move in top_moves:
        b.push(move)
        board_value = minimax_alphabeta(b, depth - 1, alpha, beta, not is_maximizing, no_top_moves ,move, prune=prune)
        b.pop()

        if (is_maximizing and board_value > best_value) or (not is_maximizing and board_value < best_value):
            best_value = board_value
            best_move = move
    return best_move

def uci(msg: str):
    """Returns result of UCI protocol given passed message"""
    if msg == "uci":
        print("id name Board Flipper Chess Bot")
        print("id author Ashen De Silva")
        print("uciok")
    elif msg == "isready":
        print("readyok")
    elif msg == "show":
        print(board)
    elif msg == "moves":
        print(board.legal_moves)
    elif msg.startswith("position startpos moves"):
        board.clear()
        board.set_fen(chess.STARTING_FEN)
        moves = msg.split()[3:]
        for move in moves:
            board.push(chess.Move.from_uci(move))
    elif msg.startswith("position fen"):
        fen = msg.removeprefix("position fen ")
        board.set_fen(fen)
    elif msg.startswith("go"):
        move = make_them_flip_the_board(board, MAX_DEPTH, NO_TOP_MOVES)
        print(f"bestmove {move}")
    elif msg == "quit":
        sys.exit(0)
    return

def main():
    """
    Main function for handling UCI input or visualizing a predefined position.

    If run with the "draw" argument, displays a visualization of the board and decision tree.
    Otherwise, continuously listens for UCI commands.

    Returns:
    - None
    """
    try:
        while True:
            uci(input())
    except Exception as e:
        print(F"Fatal Error: {e}")
        raise

if __name__ == "__main__":
    main()
