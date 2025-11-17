# Dependencies:
# - pandas
# - numpy
# - itertools (via pandas merge for pair generation)
#
# How to test this module:
# 1. Launch a Python shell from the project root.
# 2. Load the raw data (`from src.data_loading import load_players, load_player_regular_season_career, load_team_season, load_teams`).
# 3. Call the functions defined here to verify the returned feature tables (e.g., `build_player_feature_table(...)`).

from __future__ import annotations

import numpy as np
import pandas as pd


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with stripped, lowercase column names for predictable access."""
    new_df = df.copy()
    new_df.columns = [col.strip().lower() for col in new_df.columns]
    return new_df


def _safe_ratio(numerator: pd.Series, denominator: pd.Series, fill_value: float = 0.0) -> pd.Series:
    """Element-wise division that guards against division by zero."""
    ratio = numerator.astype(float) / denominator.replace({0: np.nan})
    return ratio.fillna(fill_value)


def build_player_feature_table(
    players_df: pd.DataFrame,
    reg_career_df: pd.DataFrame,
    playoff_career_df: pd.DataFrame | None = None,
    allstar_df: pd.DataFrame | None = None,
    min_games: int = 82,
) -> pd.DataFrame:
    """Create a consolidated player feature table suitable for outlier detection."""
    players_df = _standardize_columns(players_df)
    reg_career_df = _standardize_columns(reg_career_df)

    required_stats = {"ilkid", "gp", "minutes", "pts", "reb", "asts", "stl", "blk", "turnover", "fga", "fgm", "fta", "ftm", "tpa", "tpm"}
    missing = required_stats - set(reg_career_df.columns)
    if missing:
        raise ValueError(f"Regular season career data missing columns: {missing}")

    player_meta_cols = {
        "firstname": "bio_firstname",
        "lastname": "bio_lastname",
        "position": "position",
        "firstseason": "first_season",
        "lastseason": "last_season",
    }
    player_meta = players_df[["ilkid", *player_meta_cols.keys()]].rename(columns=player_meta_cols)

    feature_df = reg_career_df.merge(player_meta, on="ilkid", how="left")
    feature_df["player_name"] = (
        feature_df["bio_firstname"].fillna("").str.title().str.strip()
        + " "
        + feature_df["bio_lastname"].fillna("").str.title().str.strip()
    ).str.strip()

    # Filtering out players below the minimum games threshold keeps tiny sample sizes from distorting outlier detection.
    feature_df = feature_df[feature_df["gp"] >= min_games].copy()
    feature_df["minutes"] = feature_df["minutes"].replace({0: np.nan})

    if playoff_career_df is not None and not playoff_career_df.empty:
        playoff_df = _standardize_columns(playoff_career_df)
        rename_map = {col: f"playoff_{col}" for col in playoff_df.columns if col not in {"ilkid"}}
        playoff_df = playoff_df.rename(columns=rename_map)
        feature_df = feature_df.merge(playoff_df, on="ilkid", how="left")
        # Capture playoff scoring rates to distinguish players who elevate their game.
        feature_df["playoff_ppg"] = _safe_ratio(feature_df["playoff_pts"], feature_df["playoff_gp"])
        feature_df["playoff_rpg"] = _safe_ratio(feature_df["playoff_reb"], feature_df["playoff_gp"])
    else:
        feature_df["playoff_gp"] = 0
        feature_df["playoff_ppg"] = 0.0
        feature_df["playoff_rpg"] = 0.0

    if allstar_df is not None and not allstar_df.empty:
        allstar_df = _standardize_columns(allstar_df)
        agg = (
            allstar_df.groupby("ilkid")
            .agg(
                allstar_appearances=("year", "nunique"),
                allstar_games=("gp", "sum"),
                allstar_minutes=("minutes", "sum"),
                allstar_points=("pts", "sum"),
            )
            .reset_index()
        )
        feature_df = feature_df.merge(agg, on="ilkid", how="left")
    else:
        feature_df["allstar_appearances"] = 0
        feature_df["allstar_games"] = 0
        feature_df["allstar_minutes"] = 0
        feature_df["allstar_points"] = 0

    per_game_map = {"pts": "ppg", "reb": "rpg", "asts": "apg", "stl": "spg", "blk": "bpg", "turnover": "tpg"}
    for raw_col, feature_name in per_game_map.items():
        feature_df[feature_name] = _safe_ratio(feature_df[raw_col], feature_df["gp"])

    per36_base = ["pts", "reb", "asts"]
    # Per-minute (per-36) metrics allow fairer comparisons between starters and bench players who log different minutes.
    for raw_col in per36_base:
        feature_df[f"{raw_col}_per_36"] = _safe_ratio(feature_df[raw_col], feature_df["minutes"]).mul(36).fillna(0.0)

    # Shooting efficiencies capture scoring quality, not just volume.
    feature_df["fg_pct"] = _safe_ratio(feature_df["fgm"], feature_df["fga"])
    feature_df["ft_pct"] = _safe_ratio(feature_df["ftm"], feature_df["fta"])
    feature_df["three_pct"] = _safe_ratio(feature_df["tpm"], feature_df["tpa"])

    feature_df["assist_to_turnover"] = _safe_ratio(feature_df["asts"], feature_df["turnover"])
    feature_df["impact_score"] = feature_df["ppg"] + 0.7 * feature_df["rpg"] + 1.2 * feature_df["apg"]
    feature_df["usage_proxy"] = _safe_ratio(feature_df["fga"] + feature_df["fta"] * 0.44, feature_df["minutes"]).mul(36).fillna(0.0)

    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
    feature_df[numeric_cols] = feature_df[numeric_cols].fillna(0.0)

    ordered_cols = [
        "ilkid",
        "player_name",
        "position",
        "first_season",
        "last_season",
        "gp",
        "minutes",
        *per_game_map.values(),
        *(f"{col}_per_36" for col in per36_base),
        "fg_pct",
        "ft_pct",
        "three_pct",
        "assist_to_turnover",
        "impact_score",
        "usage_proxy",
        "playoff_gp",
        "playoff_ppg",
        "allstar_appearances",
    ]
    # Include any columns we created beyond the ordered set.
    ordered_cols = [col for col in ordered_cols if col in feature_df.columns]
    remaining_cols = [col for col in feature_df.columns if col not in ordered_cols]
    return feature_df[ordered_cols + remaining_cols]


def build_team_season_features(team_season_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
    """Generate engineered metrics per team-season."""
    team_season_df = _standardize_columns(team_season_df)
    teams_df = _standardize_columns(teams_df)

    required = {"team", "year", "won", "lost", "o_pts", "d_pts", "pace", "o_asts", "o_to", "o_reb", "d_reb"}
    missing = required - set(team_season_df.columns)
    if missing:
        raise ValueError(f"team_season_df missing columns: {missing}")

    feat_df = team_season_df.copy()
    feat_df["games"] = feat_df["won"] + feat_df["lost"]
    feat_df = feat_df[feat_df["games"] > 0].copy()
    feat_df["win_pct"] = _safe_ratio(feat_df["won"], feat_df["games"])
    feat_df["off_ppg"] = _safe_ratio(feat_df["o_pts"], feat_df["games"])
    feat_df["def_ppg"] = _safe_ratio(feat_df["d_pts"], feat_df["games"])
    feat_df["margin"] = feat_df["off_ppg"] - feat_df["def_ppg"]
    feat_df["off_rating"] = _safe_ratio(feat_df["o_pts"], feat_df["pace"])
    feat_df["def_rating"] = _safe_ratio(feat_df["d_pts"], feat_df["pace"])
    feat_df["turnover_rate"] = _safe_ratio(feat_df["o_to"], feat_df["games"])
    feat_df["assist_rate"] = _safe_ratio(feat_df["o_asts"], feat_df["games"])
    feat_df["rebound_rate"] = _safe_ratio(feat_df["o_reb"], feat_df["games"])
    feat_df["opp_rebound_rate"] = _safe_ratio(feat_df["d_reb"], feat_df["games"])
    feat_df["pace"] = feat_df["pace"].fillna(feat_df["pace"].median())

    teams_meta = teams_df[["team", "location", "name"]].copy()
    teams_meta["team_name"] = (teams_meta["location"].str.title().str.strip() + " " + teams_meta["name"].str.title().str.strip()).str.strip()

    feat_df = feat_df.merge(teams_meta[["team", "team_name"]], on="team", how="left")
    feat_df["team_name"] = feat_df["team_name"].fillna(feat_df["team"])

    return feat_df


def build_pairwise_matchups(team_features_df: pd.DataFrame) -> pd.DataFrame:
    """Create a pairwise team-versus-team dataset with difference features and binary outcomes."""
    team_features_df = _standardize_columns(team_features_df)
    required_cols = {"team", "year", "team_name", "win_pct", "off_ppg", "def_ppg", "margin", "off_rating", "def_rating", "turnover_rate", "rebound_rate", "pace"}
    missing = required_cols - set(team_features_df.columns)
    if missing:
        raise ValueError(f"team_features_df missing columns: {missing}")

    pair_frames: list[pd.DataFrame] = []
    for _, group in team_features_df.groupby("year"):
        if len(group) < 2:
            continue
        pairs = group.merge(group, on="year", suffixes=("_a", "_b"))
        pairs = pairs[pairs["team_a"] != pairs["team_b"]].copy()
        if pairs.empty:
            continue

        win_diff = pairs["win_pct_a"] - pairs["win_pct_b"]
        margin_diff = pairs["margin_a"] - pairs["margin_b"]
        # Label indicates whether Team A is expected to beat Team B given season-long strength.
        pairs["label"] = np.where(win_diff > 0, 1, np.where(win_diff < 0, 0, np.where(margin_diff > 0, 1, 0)))

        comparison_features = ["win_pct", "off_ppg", "def_ppg", "margin", "off_rating", "def_rating", "turnover_rate", "rebound_rate", "pace"]
        # Contrastive (Team A minus Team B) features encode the matchup context rather than absolute quality.
        for feat in comparison_features:
            pairs[f"diff_{feat}"] = pairs[f"{feat}_a"] - pairs[f"{feat}_b"]

        pair_frames.append(
            pairs[
                [
                    "year",
                    "team_a",
                    "team_name_a",
                    "team_b",
                    "team_name_b",
                    "label",
                    *[f"diff_{feat}" for feat in comparison_features],
                ]
            ]
        )

    if not pair_frames:
        return pd.DataFrame(
            columns=[
                "year",
                "team_a",
                "team_name_a",
                "team_b",
                "team_name_b",
                "label",
                "diff_win_pct",
                "diff_off_ppg",
                "diff_def_ppg",
                "diff_margin",
                "diff_off_rating",
                "diff_def_rating",
                "diff_turnover_rate",
                "diff_rebound_rate",
                "diff_pace",
            ]
        )

    pair_df = pd.concat(pair_frames, ignore_index=True)
    pair_df = pair_df.sort_values(["year", "team_a", "team_b"]).reset_index(drop=True)
    return pair_df
