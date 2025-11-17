"""Regenerate the project notebooks with consistent structure and instructional content."""
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


def write_notebook(path: Path, cells: list) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nbf.write(nb, path)


def notebook_one_cells() -> list:
    cells = [
        nbf.v4.new_markdown_cell(
            dedent(
                """# Notebook 01 – Exploratory Analysis & Outlier Detection

**Dependencies:** pandas, numpy, seaborn, matplotlib, scikit-learn, scipy.

We explore the basketballreference.com snapshot and surface outstanding players via robust anomaly detection."""
            )
        ),
        nbf.v4.new_markdown_cell(
            dedent(
                """## How to run this notebook
1. Install dependencies with `pip install -r requirements.txt`.
2. Launch Jupyter/Lab from the project root so `src.*` modules resolve.
3. Execute cells sequentially; plots and tables will be saved under `figures/` and `tables/`.
4. Rerun after tweaking feature-engineering or modeling logic to refresh the outputs."""
            )
        ),
        nbf.v4.new_markdown_cell("## Setup"),
        nbf.v4.new_code_cell(
            dedent(
                """import sys
from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure the project root (parent of notebooks/) is on sys.path
PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import data_loading as dl
from src import feature_engineering as fe
from src import models_outliers as mo

sns.set_theme(style="whitegrid")

FIG_DIR = PROJECT_ROOT / "figures"
TABLE_DIR = PROJECT_ROOT / "tables"
FIG_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)"""
            )
        ),
        nbf.v4.new_markdown_cell(
            dedent(
                """## Data Overview
Load the player master table plus career stat summaries. This keeps notebook logic tidy by centralising file handling inside `src.data_loading`."""
            )
        ),
        nbf.v4.new_code_cell(
            dedent(
                """players = dl.load_players()
reg_career = dl.load_player_regular_season_career()
playoff_career = dl.load_player_playoffs_career()
allstar = dl.load_player_allstar()

print(f"Players: {players.shape}, Regular-season career rows: {reg_career.shape}")
players.head()"""
            )
        ),
        nbf.v4.new_markdown_cell(
            dedent(
                """## Feature Engineering for Player Profiles
Convert raw totals into comparable per-game/per-minute stats and add postseason/all-star signals so our outlier detectors can spot genuinely elite careers."""
            )
        ),
        nbf.v4.new_code_cell(
            dedent(
                """player_features = fe.build_player_feature_table(
    players_df=players,
    reg_career_df=reg_career,
    playoff_career_df=playoff_career,
    allstar_df=allstar,
    min_games=82,
)

identifier_cols = [
    "ilkid",
    "player_name",
    "position",
    "impact_score",
    "ppg",
    "apg",
    "rpg",
    "fg_pct",
    "usage_proxy",
    "playoff_ppg",
    "allstar_appearances",
]

player_features[identifier_cols + ["gp"]].head()"""
            )
        ),
        nbf.v4.new_markdown_cell(
            dedent(
                """## Distributional EDA
Visualise scoring, playmaking, and overall impact to understand how heavy the tails are before flagging anomalies."""
            )
        ),
        nbf.v4.new_code_cell(
            dedent(
                """metrics_to_plot = ["ppg", "apg", "impact_score", "usage_proxy"]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.ravel()
for metric, ax in zip(metrics_to_plot, axes):
    sns.histplot(player_features[metric], bins=40, ax=ax, color="#3b8bc4")
    ax.set_title(f"Distribution of {metric}")
    ax.axvline(player_features[metric].median(), color="black", linestyle="--", label="Median")
    ax.legend()
fig.suptitle("Key Player Metric Distributions", fontsize=14)
fig.tight_layout()
fig.savefig(FIG_DIR / "player_metric_distributions.png", dpi=300)
fig"""
            )
        ),
        nbf.v4.new_markdown_cell(
            dedent(
                """## Correlation Heatmap
Highlight which stats move together; this informs how redundant the feature space is before anomaly detection."""
            )
        ),
        nbf.v4.new_code_cell(
            dedent(
                """corr_features = ["ppg", "apg", "rpg", "impact_score", "usage_proxy", "assist_to_turnover", "fg_pct"]
corr = player_features[corr_features].corr()
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
ax.set_title("Correlation Matrix of Core Features")
fig.tight_layout()
fig.savefig(FIG_DIR / "player_metric_correlation.png", dpi=300)
fig"""
            )
        ),
        nbf.v4.new_markdown_cell(
            dedent(
                """## MAD-Based Outstanding Player Detection
Median absolute deviation (MAD) is resistant to extreme values, making it ideal for surfacing players far beyond the norm on efficiency/impact axes."""
            )
        ),
        nbf.v4.new_code_cell(
            dedent(
                """mad_features = ["ppg", "apg", "rpg", "impact_score", "usage_proxy", "assist_to_turnover", "fg_pct"]
mad_result = mo.detect_outliers_via_mad(player_features, mad_features, threshold=3.0)
top_mad = mo.summarize_top_outliers(mad_result, identifier_cols, top_n=15)
mad_path = TABLE_DIR / "top_outliers_mad.csv"
top_mad.to_csv(mad_path, index=False)
print(f"Saved MAD outliers table to {mad_path}")
top_mad"""
            )
        ),
        nbf.v4.new_markdown_cell(
            dedent(
                """## Isolation Forest Detection
IsolationForest inspects the multivariate space, catching players whose joint profile (scoring + playmaking + playoff excellence) is rare."""
            )
        ),
        nbf.v4.new_code_cell(
            dedent(
                """iso_features = [
    "impact_score",
    "ppg",
    "apg",
    "rpg",
    "usage_proxy",
    "assist_to_turnover",
    "fg_pct",
    "playoff_ppg",
    "allstar_appearances",
]
iforest_result = mo.run_isolation_forest(player_features, iso_features, contamination=0.03)
top_iforest = mo.summarize_top_outliers(iforest_result, identifier_cols, top_n=15, ascending=True)
iforest_path = TABLE_DIR / "top_outliers_iforest.csv"
top_iforest.to_csv(iforest_path, index=False)
print(f"Saved IsolationForest table to {iforest_path}")
top_iforest"""
            )
        ),
        nbf.v4.new_markdown_cell(
            dedent(
                """## Method Overlap
Compare MAD vs. IsolationForest selections to see which stars appear consistently."""
            )
        ),
        nbf.v4.new_code_cell(
            dedent(
                """mad_ids = set(top_mad["ilkid"])
iforest_ids = set(top_iforest["ilkid"])
overlap_ids = sorted(mad_ids & iforest_ids)
print(f"Players highlighted by both methods: {len(overlap_ids)}")
overlap_df = player_features[player_features["ilkid"].isin(overlap_ids)][identifier_cols]
overlap_path = TABLE_DIR / "outlier_overlap.csv"
overlap_df.to_csv(overlap_path, index=False)
overlap_df"""
            )
        ),
        nbf.v4.new_markdown_cell(
            dedent(
                """## Visualise Flagged Players
Plot impact vs. usage to show why certain players stand apart; annotate those flagged by the MAD approach."""
            )
        ),
        nbf.v4.new_code_cell(
            dedent(
                """plot_df = mad_result.dataframe
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(
    data=plot_df,
    x="usage_proxy",
    y="impact_score",
    hue="is_outlier_mad",
    palette={True: "#d73027", False: "#4575b4"},
    ax=ax,
)
ax.set_title("Usage vs. Impact with MAD Outliers Highlighted")
ax.set_xlabel("Usage Proxy (per 36 possessions)")
ax.set_ylabel("Impact Score (ppg + weighted rebounds/assists)")
for _, row in top_mad.iterrows():
    ax.text(row["usage_proxy"], row["impact_score"], row["player_name"], fontsize=8)
fig.tight_layout()
scatter_path = FIG_DIR / "usage_vs_impact_outliers.png"
fig.savefig(scatter_path, dpi=300)
print(f"Saved scatter plot to {scatter_path}")
fig"""
            )
        ),
        nbf.v4.new_markdown_cell(
            dedent(
                """## Takeaways
- MAD and IsolationForest largely agree on the most dominant multi-dimensional players, lending credibility to the outstanding-player list.
- The helper tables/figures can be referenced directly in the written report (Results & Discussion section)."""
            )
        ),
    ]
    return cells


def notebook_two_cells() -> list:
    cells = [
        nbf.v4.new_markdown_cell(
            dedent(
                """# Notebook 02 – Game Outcome Modelling

**Dependencies:** pandas, numpy, seaborn, matplotlib, scikit-learn."""
            )
        ),
        nbf.v4.new_markdown_cell(
            dedent(
                """## How to run this notebook
1. Install dependencies via `pip install -r requirements.txt`.
2. Start Jupyter from the project root for clean relative paths.
3. Execute cells in order; this will regenerate engineered team features, matchup datasets, models, and plots saved to `figures/` & `tables/`.
4. Use the outputs in the report’s Methods/Results sections."""
            )
        ),
        nbf.v4.new_markdown_cell("## Setup"),
        nbf.v4.new_code_cell(
            dedent(
                """import sys
from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

# Ensure src/ is importable whether notebook runs from root or notebooks/
PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import data_loading as dl
from src import feature_engineering as fe
from src import models_game_outcome as mgo
from src import evaluation as eval_utils

sns.set_theme(style="whitegrid")

FIG_DIR = PROJECT_ROOT / "figures"
TABLE_DIR = PROJECT_ROOT / "tables"
FIG_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)"""
            )
        ),
        nbf.v4.new_markdown_cell("## Load and Engineer Team Features"),
        nbf.v4.new_code_cell(
            dedent(
                """team_season = dl.load_team_season()
teams = dl.load_teams()
team_features = fe.build_team_season_features(team_season, teams)
print(f\"Team features: {team_features.shape}\")
team_features.head()"""
            )
        ),
        nbf.v4.new_markdown_cell("## Construct Pairwise Matchups"),
        nbf.v4.new_code_cell(
            dedent(
                """matchups = fe.build_pairwise_matchups(team_features)
print(f\"Matchups dataset: {matchups.shape}\")
matchups.head()"""
            )
        ),
        nbf.v4.new_markdown_cell("## Train/Test Split (Season-Aware)"),
        nbf.v4.new_code_cell(
            dedent(
                """train_df, test_df = mgo.season_train_test_split(matchups, train_ratio=0.7)
train_seasons = sorted(train_df['year'].unique())
test_seasons = sorted(test_df['year'].unique())
print(f\"Train seasons: {train_seasons[:3]} ... {train_seasons[-3:]}\")
print(f\"Test seasons: {test_seasons}\")

X_train, y_train, feature_cols = mgo.prepare_features(train_df)
X_test = test_df[feature_cols]
y_test = test_df['label']"""
            )
        ),
        nbf.v4.new_markdown_cell("## Train Baseline and Advanced Models"),
        nbf.v4.new_code_cell(
            dedent(
                """models = mgo.train_models(train_df, feature_cols=feature_cols)
list(models.keys())"""
            )
        ),
        nbf.v4.new_markdown_cell("## Evaluate Models on Held-Out Seasons"),
        nbf.v4.new_code_cell(
            dedent(
                """evaluation_results = {}
for name, model in models.items():
    res = eval_utils.evaluate_classifier(model, X_test, y_test)
    evaluation_results[name] = res

metrics_df = eval_utils.metrics_to_dataframe(evaluation_results)
metrics_path = TABLE_DIR / "model_comparison.csv"
eval_utils.save_metrics_table(metrics_df, metrics_path)
print(f"Saved model metrics to {metrics_path}")
metrics_df"""
            )
        ),
        nbf.v4.new_markdown_cell("## Confusion Matrices"),
        nbf.v4.new_code_cell(
            dedent(
                """for name in ["logistic_regression", "gradient_boosting"]:
    res = evaluation_results[name]
    cf_path = FIG_DIR / f"confusion_{name}.png"
    eval_utils.plot_confusion_matrix(y_test, res["y_pred"], title=f"{name.title()} Confusion Matrix", save_path=cf_path)
    plt.close()
    print(f"Saved confusion matrix for {name} to {cf_path}")"""
            )
        ),
        nbf.v4.new_markdown_cell("## ROC Curves"),
        nbf.v4.new_code_cell(
            dedent(
                """fig, ax = plt.subplots(figsize=(6, 5))
for name in ["logistic_regression", "gradient_boosting", "random_forest"]:
    res = evaluation_results[name]
    fpr, tpr, _ = metrics.roc_curve(y_test, res["y_score"])
    roc_auc = metrics.auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves – Game Outcome Models")
ax.legend()
fig.tight_layout()
roc_path = FIG_DIR / "roc_models.png"
fig.savefig(roc_path, dpi=300)
print(f"Saved ROC plot to {roc_path}")
fig"""
            )
        ),
        nbf.v4.new_markdown_cell("## Feature Importance (Random Forest)"),
        nbf.v4.new_code_cell(
            dedent(
                """fi_path = FIG_DIR / "feature_importance_random_forest.png"
eval_utils.plot_feature_importances(models["random_forest"], feature_cols, top_n=10, save_path=fi_path)
plt.close()
print(f"Saved feature importance figure to {fi_path}")"""
            )
        ),
        nbf.v4.new_markdown_cell(
            dedent(
                """## Discussion
- Logistic regression provides an interpretable baseline for report narratives.
- Tree-based models capture non-linear effects; feature-importance plots reveal which season metrics drive matchup wins."""
            )
        ),
    ]
    return cells


def notebook_three_cells() -> list:
    cells = [
        nbf.v4.new_markdown_cell(
            dedent(
                """# Notebook 03 – Results Summary

**Dependencies:** pandas, numpy, matplotlib."""
            )
        ),
        nbf.v4.new_markdown_cell(
            dedent(
                """## How to run this notebook
1. Execute Notebooks 01 & 02 first so that their tables/figures exist.
2. Launch from the project root and run every cell.
3. Use the rendered tables/figures/text snippets directly in the written report."""
            )
        ),
        nbf.v4.new_markdown_cell("## Setup"),
        nbf.v4.new_code_cell(
            dedent(
                """import sys
from pathlib import Path

import pandas as pd
from IPython.display import Markdown, display

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FIG_DIR = PROJECT_ROOT / "figures"
TABLE_DIR = PROJECT_ROOT / "tables"
"""
            )
        ),
        nbf.v4.new_markdown_cell("## Outstanding Players Summary"),
        nbf.v4.new_code_cell(
            dedent(
                """mad_table = pd.read_csv(TABLE_DIR / "top_outliers_mad.csv")
iforest_table = pd.read_csv(TABLE_DIR / "top_outliers_iforest.csv")
overlap_table = pd.read_csv(TABLE_DIR / "outlier_overlap.csv")

display(Markdown("**Top MAD Outliers**"))
display(mad_table.head(10))

display(Markdown("**Top IsolationForest Outliers**"))
display(iforest_table.head(10))

display(Markdown("**Consensus Outstanding Players**"))
display(overlap_table)"""
            )
        ),
        nbf.v4.new_markdown_cell(
            dedent(
                """## Narrative Notes for Report
- MAD highlights classic statistical monsters (high usage and efficiency).
- IsolationForest surfaces balanced contributors with playoff/all-star credentials.
- Overlap table underpins the outstanding-player discussion."""
            )
        ),
        nbf.v4.new_markdown_cell("## Game Outcome Model Summary"),
        nbf.v4.new_code_cell(
            dedent(
                """model_metrics = pd.read_csv(TABLE_DIR / "model_comparison.csv", index_col="model")
display(model_metrics)"""
            )
        ),
        nbf.v4.new_markdown_cell(
            dedent(
                """## Suggested Report Text
- Logistic regression serves as a transparent baseline with accuracy reported above.
- Gradient boosting/random forest improve ROC-AUC by capturing interaction effects among pace, margin, and turnover features.
- Confusion matrices (`figures/confusion_*.png`) and ROC curves (`figures/roc_models.png`) can be referenced as Figures in the report."""
            )
        ),
    ]
    return cells


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    nb_dir = root / "notebooks"
    nb_dir.mkdir(exist_ok=True)

    write_notebook(nb_dir / "01_eda_and_outlier_detection.ipynb", notebook_one_cells())
    write_notebook(nb_dir / "02_game_outcome_modelling.ipynb", notebook_two_cells())
    write_notebook(nb_dir / "03_results_summary.ipynb", notebook_three_cells())


if __name__ == "__main__":
    main()
