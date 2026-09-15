"""Regenerate data/reference/players.csv from cards_seed.csv."""

import pandas as pd

if __name__ == "__main__":
    cards = pd.read_csv("data/reference/cards_seed.csv")
    players = cards[["mlb_id", "player_name", "role"]].rename(columns={"player_name": "name"})
    players.to_csv("data/reference/players.csv", index=False)
    print(f"wrote {len(players)} players")
