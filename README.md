# NBA Prediction Project

Machine-learning exploration for COMP721 focused on the 2004–2005 Basketball Reference snapshot (`databasebasketball/`). We identify outstanding players via robust outlier detection and build models that estimate game outcomes between any two teams based on season-level statistics. All analysis runs locally; no external data downloads are required.

## Repository Layout

- `databasebasketball/`: raw text files from basketballreference.com (players, teams, seasons, playoffs, etc.).
- `src/`: reusable Python modules (data loading, feature engineering, modeling, evaluation).
- `notebooks/`: Jupyter notebooks used for EDA, modeling, and report-ready summaries.
- `figures/` & `tables/`: auto-generated outputs saved by the notebooks (PNG plots, CSV/Markdown tables).
- `requirements.txt`: Python dependencies.

## Environment Setup

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter for interactive work:
   ```bash
   jupyter notebook
   ```

## Running the Project

1. **EDA & Outlier Detection**  
   Open `notebooks/01_eda_and_outlier_detection.ipynb`, run cells sequentially. This loads player data via `src.data_loading`, engineers features via `src.feature_engineering`, fits outlier detectors, and writes figures/tables.

2. **Game Outcome Modeling**  
   Use `notebooks/02_game_outcome_modelling.ipynb` to generate team-level features, construct matchup datasets, train/evaluate multiple classifiers, and save evaluation artifacts.

3. **Results Summary**  
   Execute `notebooks/03_results_summary.ipynb` to pull saved tables/plots into report-ready summaries and narratives.

All notebooks include a “Dependencies” and “How to run” section for reproducibility. For scripted experimentation, import the utilities directly, e.g.:

```python
from src.data_loading import load_team_season, load_teams
from src.feature_engineering import build_team_season_features
```

## Notes

- All datasets must be accessed via the provided `databasebasketball` folder; internet fetching is disabled by design.
- Cite basketballreference.com in any derivative work, per the dataset’s license.
