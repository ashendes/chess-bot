# Chess Carnage: Learning-Based Chess Evaluation ♔♕♗♘♙♖

Train regression models to evaluate chess positions, integrate the learned evaluator into a UCI chess bot, and compare model performance through automated tournaments and analysis.

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
- Model files (`.pkl`) are saved in the repository root.