import csv
import chess
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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

def load_dataset(filename):
    """
    Loads the dataset from a CSV file.
    """
    x = []  # Features
    y = []  # Targets
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            fen = row['fen']
            evaluation = float(row['eval'])
            features = fen_to_features(fen)
            x.append(features)
            y.append(evaluation)
    return x, y

def train_model(x_train, y_train):
    """
    Trains a linear regression model using the provided training data.
    """
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model

def evaluate_model(model, x_test, y_test):
    """
    Evaluates the trained model on the test data and prints the mean squared error.
    """
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

def evaluate_position_with_model(fen, model):
    """
    Evaluates a position using the learned model.
    """
    features = fen_to_features(fen)
    evaluation = model.predict([features])
    return evaluation[0]

def main():
    # Load dataset
    x, y = load_dataset('chess_evaluations.csv')

    # Split dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train model
    model = train_model(x_train, y_train)

    # Evaluate model
    evaluate_model(model, x_test, y_test)

    # Test the model with a FEN string
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    evaluation = evaluate_position_with_model(fen, model)
    print(f"Evaluation for FEN '{fen}': {evaluation}")

if __name__ == "__main__":
    main()
