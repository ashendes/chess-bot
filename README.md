# Chess Carnage: Learning-Based Chess Evaluation ♔♕♗♘♙♖

This project explores learning-based chess evaluation by training regression models to score positions and using the learned evaluator inside a UCI-compatible bot. The purpose is to build a practical pipeline from engine-labeled FEN data to an evaluation function, then measure how different models affect bot strength and decision quality through automated tournaments and analysis. The workflow generates labeled FEN positions from Stockfish, trains a regression model, then plugs the evaluator into a minimax search for UCI play and benchmarking.

## Tech Stack

- Python 3
- `python-chess`
- `scikit-learn`
- `numpy`, `matplotlib` (analysis notebook)
- `chester` (optional, for `tournament.py`)
- Stockfish (for data generation)

## Setup

1. Install dependencies:
   - `pip install python-chess scikit-learn numpy matplotlib`
   - `pip install chester` (optional, only required for `tournament.py`)
2. Install Stockfish and set the engine path:
   - `export STOCKFISH_PATH="/path/to/stockfish"`

## How to Run

1. Generate training data:
   - `python fen_generator.py`
2. Train a model and persist it to disk:
   - `python learner.py`
3. Run the learned UCI bot:
   - `python board_flipper_chess_bot.py`
4. Run the baseline random UCI bot:
   - `python random_chess_bot.py`
5. Run a tournament (optional):
   - `python tournament.py`
   - Update the `players` list in `tournament.py` to point to your engines if needed.

## Analysis

The notebook `chess-carnage-analysis.ipynb` compares regression models, tournament results, and performance metrics with visualizations.

## Project Artifacts

- `tournament_results/` contains tournament summaries.
- `models-trained/` contains trained model files (`.pkl`).