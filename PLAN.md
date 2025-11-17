# Data Preparation and Modeling Plan

## A. Player-Level Outlier Detection

- **Sources**: `players.txt` (bio data), `player_regular_season_career.txt` (career totals), `player_playoffs_career.txt` (optional playoff totals), `player_allstar.txt` (optional all-star counts).
- **Linking Keys**: Join on `ilkid` to align player metadata with career statistics; concatenate `firstname`/`lastname` for reporting fields.
- **Preprocessing Pipeline**:
  1. Clean column names (lowercase, snake case) and enforce numeric dtypes for stat columns.
  2. Filter to players with at least 82 career games (≈ one full season) or >1,200 minutes to remove tiny samples.
  3. Handle missing stat totals via zeros or imputed averages when safe; drop rows lacking essential identifiers.
- **Feature Engineering**:
  - Compute per-game metrics (`ppg`, `rpg`, `apg`, `spg`, `bpg`, `tpg`) and possessions-adjusted rates (per 36 minutes when minutes > 0).
  - Shooting efficiencies: `fg_pct = fgm / fga`, `ft_pct = ftm / fta`, `three_pct = tpm / tpa` with safe division.
  - Composite indices: e.g., `impact_score = ppg + 0.7 * rpg + 1.2 * apg`, assist-to-turnover ratio, rebound splits.
  - Merge playoff and all-star indicators (e.g., playoff minutes per game, number of all-star appearances) to capture postseason excellence.
- **Outlier Detection Methods**:
  1. **Robust Z/MAD Analysis** on selected efficiency and per-minute metrics to highlight players several MADs above the median.
  2. **IsolationForest** (multivariate) on standardized feature vectors (per-game, efficiency, playoff indicators) to surface players whose combined profile is rare.
  3. (Optional) **Local Outlier Factor** for density-based anomalies to check consistency across techniques.
- **Outputs**: Ranked tables of top anomalies per method, overlap analysis, and contextual stats saved to `tables/top_players_...csv`; scatter/distribution plots saved to `figures/`.

## B. Team-Level Game Outcome Prediction

- **Sources**: `team_season.txt` (season totals), `teams.txt` (team metadata for readable names).
- **Team Feature Construction**:
  - Normalize totals by 82 games to derive per-game offense (`off_ppg = o_pts / games`), defense (`def_ppg = d_pts / games`), rebounding, assist, turnover rates.
  - Compute win percentage, pace-adjusted efficiency proxies (e.g., `off_rating = o_pts / pace`, `def_rating = d_pts / pace` when pace > 0).
  - Derived statistics: margin of victory (`off_ppg - def_ppg`), shooting efficiencies (`o_fgm / o_fga`, etc.), possession control metrics (rebounds per opponent rebound, turnover rate).
- **Pairwise Matchups**:
  1. For each season, take all ordered combinations of distinct teams (A vs B).
  2. Define label `y = 1` if Team A win% > Team B win% (tie-breaking with margin of victory); `y = 0` otherwise.
  3. Create difference features: `diff_win_pct = win_pct_A - win_pct_B`, `diff_off_ppg`, `diff_margin`, `diff_turnover_rate`, etc.; include optional ratios for interpretability.
  4. Record identifiers (`team_A`, `team_B`, `season`) for back-referencing and analysis.
- **Modeling Strategy**:
  - Split data with season-aware approach (e.g., earliest 70% seasons for training, remaining for test) to simulate forecasting new seasons.
  - Baseline: Logistic Regression (interpretable coefficients) using standardized features.
  - Advanced: Random Forest or Gradient Boosting (captures non-linear interactions) with feature importance plots.
  - Optional stacked ensemble (e.g., logistic on top of RF + Gradient Boosting predictions) if time allows.
  - Evaluate via accuracy, precision, recall, F1, ROC-AUC, confusion matrices, and calibration analysis.
- **Artifacts**: Save model comparison table to `tables/model_comparison.csv`, confusion matrix/ROC plots to `figures/`, and document assumptions for report integration.
