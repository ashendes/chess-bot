#!/usr/bin/env python
import chess
import chess.svg
import random
import sys
import networkx as nx
import pickle

MAX_DEPTH = 7
NO_TOP_MOVES = 8
board = chess.Board()
G = nx.DiGraph()
G.add_node("", utility=None)

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

def evaluate_board(b: chess.Board):
    """
    Evaluate the chess board position based on material value and piece position.

    Parameters:
    - b (chess.Board): The current chess board position.

    Returns:
    - int: An integer representing the evaluation score of the board.
           Positive values favor White, and negative values favor Black.
    """
    if b.is_checkmate():
        if b.turn:
            return -100000  # Black wins
        else:
            return 100000   # White wins
    elif b.is_stalemate() or b.is_insufficient_material():
        return 0  # Draw

    value = 0
    for piece_type, piece_value in piece_values.items():
        value += len(b.pieces(piece_type, chess.WHITE)) * piece_value
        value -= len(b.pieces(piece_type, chess.BLACK)) * piece_value
        piece_table = piece_tables.get(piece_type)
        value += sum([piece_table[i] for i in b.pieces(piece_type, chess.WHITE)])
        value -= sum([piece_table[chess.square_mirror(i)] for i in b.pieces(piece_type, chess.BLACK)])

    return value

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

def minimax_alphabeta_graph(G: nx.DiGraph, board: chess.Board, depth, alpha, beta, is_maximizing, no_top_moves,
                            parent_node=None, move=None, prune=True):
    """
    Perform a minimax search with alpha-beta pruning, constructing a graph of board positions and returns the best
    move based on the evaluation scores.

    Parameters:
    - G (nx.DiGraph): The directed graph used to represent the game tree.
    - board (chess.Board): The current chess board position.
    - depth (int): The remaining depth of the search.
    - alpha (float): The alpha value for alpha-beta pruning.
    - beta (float): The beta value for alpha-beta pruning.
    - is_maximizing (bool): True if maximizing player's turn (White), False if minimizing player's turn (Black).
    - no_top_moves (int): Number of top moves to consider at each level.
    - parent_node (str): The label of the parent node in the graph.
    - move (chess.Move): The move leading to this board position.
    - prune (bool): If True, use alpha-beta pruning; otherwise, perform a regular minimax search.

    Returns:
    - int: The selected evaluation score for the board position.
    """
    node_label = str(board)
    G.add_node(node_label, utility=None)
    if parent_node is not None and move is not None:
        G.add_edge(parent_node, node_label, label=move)

    if depth == 0 or board.is_game_over():
        utility = evaluate_board(board)
        G.nodes[node_label]['utility'] = utility
        return utility

    top_moves = get_top_moves(board, is_maximizing, no_top_moves)

    best_eval = -float("inf") if is_maximizing else float("inf")
    for move in top_moves:
        board.push(move)
        evaluation = minimax_alphabeta_graph(G, board, depth - 1, alpha, beta, not is_maximizing, no_top_moves,
                                             node_label, move)
        board.pop()
        if is_maximizing:
            best_eval = max(best_eval, evaluation)
            alpha = max(alpha, best_eval) if prune else alpha
        else:
            best_eval = min(best_eval, evaluation)
            beta = min(beta, best_eval) if prune else beta
        if alpha >= beta:
            break
    G.nodes[node_label]['utility'] = best_eval
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
    G = nx.DiGraph()
    node_label = str(b)
    G.add_node(node_label, utility=None)

    is_maximizing = b.turn
    best_move = None
    best_value = -float("inf") if is_maximizing else float('inf')
    alpha = -float("inf")
    beta = float("inf")

    top_moves = get_top_moves(b, is_maximizing, no_top_moves)

    for move in top_moves:
        b.push(move)
        board_value = minimax_alphabeta_graph(G, b, depth - 1, alpha, beta, not is_maximizing, no_top_moves, node_label,
                                              move, prune=prune)
        b.pop()

        if (is_maximizing and board_value > best_value) or (not is_maximizing and board_value < best_value):
            best_value = board_value
            best_move = move
    G.nodes[node_label]['utility'] = best_value
    return best_move, G


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
        move, G = make_them_flip_the_board(board, MAX_DEPTH, NO_TOP_MOVES)
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
    if len(sys.argv) == 2 and sys.argv[1] == "draw":
        # Four Nights Sicilian Defence FEN string
        fen = "r1bqkb1r/pp1p1ppp/2n1pn2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 3 6"
        board.set_fen(fen)
        save_board_image(board)
        create_and_visualize_top3_graph(board, 4, 3)
    else:
        try:
            while True:
                uci(input())
        except Exception as e:
            print(F"Fatal Error: {e}")
            raise

if __name__ == "__main__":
    main()
