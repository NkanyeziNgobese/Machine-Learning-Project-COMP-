# Dependencies:
# - pandas
# - pathlib
# - typing
#
# How to use this module:
# 1. Import the desired loader, e.g., `from src.data_loading import load_players`.
# 2. Call the loader to obtain a pandas DataFrame (optionally setting `debug=True` for a quick summary).
# 3. Pass the returned DataFrames into the feature-engineering utilities or notebooks for further analysis.

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "databasebasketball"


def get_data_path(filename: str) -> Path:
    """Return the Path to a file within the databasebasketball directory."""
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Expected data file '{filename}' in {DATA_DIR}")
    return path


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names (strip spaces, lowercase)."""
    df = df.copy()
    df.columns = [col.strip().lower() for col in df.columns]
    return df


def _assert_required_columns(df: pd.DataFrame, required: Iterable[str], context: str) -> None:
    """Ensure that required columns exist in the frame."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def _load_csv(
    filename: str,
    required_columns: Iterable[str],
    dtype_overrides: Optional[dict] = None,
    debug: bool = False,
) -> pd.DataFrame:
    """Generic CSV loader with consistent parsing and sanity checks."""
    path = get_data_path(filename)
    df = pd.read_csv(path, sep=None, engine="python")
    df = _standardize_columns(df)
    if dtype_overrides:
        df = df.astype(dtype_overrides)
    if df.empty:
        raise ValueError(f"{filename} appears to be empty.")
    _assert_required_columns(df, required_columns, filename)
    if debug:
        sample_cols = df.columns[:6].tolist()
        print(f"[data_loading] {filename}: {df.shape[0]} rows, columns: {sample_cols}")
    return df


def _standardize_string_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Strip whitespace and uppercase selected string columns for stable joins."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            # Team/player IDs are stored uppercase in raw files; enforcing uppercase promotes consistency.
            if col in {"team", "ilkid"}:
                df[col] = df[col].str.upper()
    return df


def load_players(debug: bool = False) -> pd.DataFrame:
    """Load the players master table."""
    df = _load_csv(
        "players.txt",
        required_columns=["ilkid", "firstname", "lastname", "position", "firstseason", "lastseason"],
        debug=debug,
    )
    df = _standardize_string_columns(df, ["ilkid", "firstname", "lastname", "position"])
    return df


def load_player_regular_season_career(debug: bool = False) -> pd.DataFrame:
    """Load career regular-season totals for players."""
    required = [
        "ilkid",
        "gp",
        "minutes",
        "pts",
        "reb",
        "asts",
        "stl",
        "blk",
        "turnover",
        "fga",
        "fgm",
        "fta",
        "ftm",
        "tpa",
        "tpm",
    ]
    df = _load_csv("player_regular_season_career.txt", required_columns=required, debug=debug)
    df = _standardize_string_columns(df, ["ilkid", "leag", "firstname", "lastname"])
    return df


def load_player_playoffs_career(debug: bool = False) -> pd.DataFrame:
    """Load career playoff totals for players."""
    required = [
        "ilkid",
        "gp",
        "minutes",
        "pts",
        "reb",
        "asts",
        "stl",
        "blk",
        "turnover",
    ]
    df = _load_csv("player_playoffs_career.txt", required_columns=required, debug=debug)
    df = _standardize_string_columns(df, ["ilkid", "leag", "firstname", "lastname"])
    return df


def load_player_allstar(debug: bool = False) -> pd.DataFrame:
    """Load All-Star game stats per player."""
    required = ["ilkid", "gp", "minutes", "pts"]
    df = _load_csv("player_allstar.txt", required_columns=required, debug=debug)
    df = _standardize_string_columns(df, ["ilkid", "firstname", "lastname", "conference", "leag"])
    return df


def load_team_season(debug: bool = False) -> pd.DataFrame:
    """Load team season totals with offensive/defensive splits."""
    required = ["team", "year", "won", "lost", "o_pts", "d_pts", "pace"]
    df = _load_csv("team_season.txt", required_columns=required, debug=debug)
    df = _standardize_string_columns(df, ["team", "leag"])
    return df


def load_teams(debug: bool = False) -> pd.DataFrame:
    """Load the team metadata table."""
    required = ["team", "location", "name", "leag"]
    df = _load_csv("teams.txt", required_columns=required, debug=debug)
    df = _standardize_string_columns(df, ["team", "leag"])
    return df
