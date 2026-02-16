"""Run a round-robin tournament between UCI engines."""

import time

from chester.timecontrol import TimeControl
from chester.tournament import play_tournament

# Each string is the name/path to an executable UCI engine.
players = ["dist/random_chess_bot", "./board_flipper_chess_bot.py"]

# Specify time and increment, both in seconds.
time_control = TimeControl(initial_time=180, increment=10)

# Play each math-up twice.
n_games = 5

# Tabulate scores at the end.
scores = {}
start_time = time.time()
match_start_time = start_time

for pgn in play_tournament(
    players,
    time_control,
    n_games=n_games,
    repeat=True,  # Each opening played twice,
):
    # Printing out the game result.
    pgn.headers["Event"] = "Model Tournament"
    pgn.headers["Site"] = "Local Machine"
    print(pgn, "\n")

    # Update scores.
    white = pgn.headers["White"]
    black = pgn.headers["Black"]
    scores.setdefault(white,0) # If bot hasn't been added start at 0.
    scores.setdefault(black,0)
    results = pgn.headers["Result"].split('-')
    scores[white] += float(eval(results[0]))
    scores[black] += float(eval(results[1]))

    match_time = time.time() - match_start_time
    print(f"Match time: {round(match_time // 60)} m {round(match_time % 60)} s\n")
    match_start_time = time.time()

for (bot,score) in scores.items():
    print(bot , ":", score)

tournament_time = time.time() - start_time
print(f"Tournament time: {round(tournament_time // 60)} m {round(tournament_time % 60)} s\n")
