"""Utility to generate Project 2.0 addon notebooks with required scaffolding."""
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


def write_notebook(path: Path, cells: list) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nbf.write(nb, path)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    nb_dir = root / "notebooks" / "addons"
    nb_dir.mkdir(parents=True, exist_ok=True)

    fig_setup = dedent(
        """\
        import sys
        from pathlib import Path

        import numpy as np
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt

        PROJECT_ROOT = Path.cwd()
        if PROJECT_ROOT.name == 'addons':
            PROJECT_ROOT = PROJECT_ROOT.parent.parent
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

        FIG_DIR = PROJECT_ROOT / 'figures'
        TABLE_DIR = PROJECT_ROOT / 'tables'
        FIG_DIR.mkdir(exist_ok=True)
        TABLE_DIR.mkdir(exist_ok=True)

        sns.set_theme(style='whitegrid')
        """
    )

    # Notebook A1
    nb1_cells = [
        nbf.v4.new_markdown_cell(
            dedent(
                """\
                # Project 2.0 ADD-ON: Autoencoder-based Player Outliers

                *Experimental notebook separate from the required COMP721 deliverables.* This workflow trains a deep autoencoder on the engineered player feature matrix to surface players whose multi-dimensional stat profiles cannot be reconstructed well by the league-wide patterns.
                """
            )
        ),
        nbf.v4.new_markdown_cell(
            dedent(
                """\
                ## Advantages and Disadvantages of this Add-on
                - **Advantages:** Captures non-linear relationships, highlights subtle standout careers beyond MAD/IsolationForest, and provides an additional anomaly score for triangulation.
                - **Disadvantages:** Requires extra dependencies (TensorFlow/Keras), hyperparameter tuning, and careful interpretation of reconstruction errors; more computationally intensive than robust z-scores.
                """
            )
        ),
        nbf.v4.new_markdown_cell(
            "## How to run\n1. Ensure requirements (including TensorFlow/Keras) are installed.\n2. Launch Jupyter from the project root to resolve `src` imports.\n3. Execute cells sequentially; outputs are saved to `figures/` and `tables/` with `autoencoder_` prefixes."
        ),
        nbf.v4.new_code_cell(
            fig_setup
            + dedent(
                """\
                from src.core import data_loading as core_dl
                from src.core import feature_engineering as core_fe
                from src.core import models_outliers as core_out
                from src.addons import autoencoder_outliers as add_auto
                """
            )
        ),
        nbf.v4.new_markdown_cell("## Load player data and engineer features"),
        nbf.v4.new_code_cell(
            dedent(
                """\
                players = core_dl.load_players()
                reg_career = core_dl.load_player_regular_season_career()
                playoff_career = core_dl.load_player_playoffs_career()
                allstar = core_dl.load_player_allstar()

                player_features = core_fe.build_player_feature_table(
                    players_df=players,
                    reg_career_df=reg_career,
                    playoff_career_df=playoff_career,
                    allstar_df=allstar,
                    min_games=82,
                )
                player_features.head()
                """
            )
        ),
        nbf.v4.new_markdown_cell("## Train autoencoder and compute anomaly scores"),
        nbf.v4.new_code_cell(
            dedent(
                """\
                numeric_cols = player_features.select_dtypes(include=[np.number]).columns
                feature_matrix = player_features[numeric_cols].fillna(0.0)

                auto_model, scaler, auto_errors = add_auto.train_autoencoder(
                    feature_matrix,
                    latent_dim=8,
                    epochs=40,
                    batch_size=128,
                )

                player_features['autoencoder_error'] = auto_errors
                key_stats = [col for col in ['impact_score', 'ppg', 'apg', 'rpg', 'fg_pct', 'usage_proxy'] if col in player_features.columns]

                top_auto = add_auto.get_top_anomalies(
                    player_features[key_stats],
                    player_features['player_name'],
                    auto_errors,
                    top_k=20,
                )

                auto_table_path = TABLE_DIR / 'top_outliers_autoencoder.csv'
                top_auto.to_csv(auto_table_path, index=False)
                print(f'Saved top autoencoder outliers table to {auto_table_path}')
                top_auto.head()
                """
            )
        ),
        nbf.v4.new_markdown_cell("## Visualise reconstruction error distribution"),
        nbf.v4.new_code_cell(
            dedent(
                """\
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.histplot(player_features['autoencoder_error'], bins=40, ax=ax, color='#0077b6')
                ax.set_title('Autoencoder Reconstruction Error Distribution')
                ax.set_xlabel('Reconstruction error (MSE)')
                ax.set_ylabel('Player count')
                fig.tight_layout()
                recon_fig_path = FIG_DIR / 'autoencoder_recon_error_hist.png'
                fig.savefig(recon_fig_path, dpi=300)
                print(f'Saved histogram to {recon_fig_path}')
                fig
                """
            )
        ),
        nbf.v4.new_markdown_cell("## Compare autoencoder errors with MAD scores"),
        nbf.v4.new_code_cell(
            dedent(
                """\
                mad_features = [col for col in ['ppg', 'apg', 'rpg', 'impact_score', 'usage_proxy', 'assist_to_turnover', 'fg_pct'] if col in player_features.columns]
                mad_result = core_out.detect_outliers_via_mad(player_features, mad_features, threshold=3.0)
                mad_df = mad_result.dataframe.copy()
                mad_df['autoencoder_error'] = player_features['autoencoder_error']

                fig, ax = plt.subplots(figsize=(6, 5))
                sns.scatterplot(data=mad_df, x='mad_score', y='autoencoder_error', ax=ax, alpha=0.5)
                ax.set_title('Autoencoder Error vs. MAD Score')
                ax.set_xlabel('MAD-based anomaly score')
                ax.set_ylabel('Autoencoder reconstruction error')
                fig.tight_layout()
                scatter_path = FIG_DIR / 'autoencoder_vs_mad_scatter.png'
                fig.savefig(scatter_path, dpi=300)
                print(f'Saved comparison scatter plot to {scatter_path}')
                fig
                """
            )
        ),
        nbf.v4.new_markdown_cell(
            dedent(
                """\
                ## Summary: Should this Add-on be Included in the Final Report?
                Autoencoder-based detection offers a compelling non-linear perspective that can complement MAD/IsolationForest, but it introduces extra dependencies and interpretability overhead. It is best presented as an optional research extension unless the assessment explicitly rewards advanced modelling techniques.
                """
            )
        ),
    ]

    write_notebook(nb_dir / "A1_autoencoder_outliers_experiments.ipynb", nb1_cells)

    # Notebook A2
    nb2_cells = [
        nbf.v4.new_markdown_cell(
            dedent(
                """\
                # Project 2.0 ADD-ON: SHAP Explainability for Matchup Models

                *Experimental notebook distinct from the required submission.* We compute SHAP values for the tree-based matchup classifier to explain how feature differences (e.g., win%, margin) drive predicted winners.
                """
            )
        ),
        nbf.v4.new_markdown_cell(
            dedent(
                """\
                ## Advantages and Disadvantages of this Add-on
                - **Advantages:** Provides global and local interpretability, supports accountability in model reporting, and highlights which features drive predictions.
                - **Disadvantages:** Adds computational overhead, depends on SHAP visualisations that require careful explanation, and may overwhelm non-technical readers.
                """
            )
        ),
        nbf.v4.new_markdown_cell(
            "## How to run\n1. Ensure SHAP is installed (`pip install shap`).\n2. Launch Jupyter from the project root.\n3. Execute cells to retrain the matchup models, compute SHAP values, and save figures."
        ),
        nbf.v4.new_code_cell(
            fig_setup
            + dedent(
                """\
                from sklearn import metrics

                from src.core import data_loading as core_dl
                from src.core import feature_engineering as core_fe
                from src.core import models_game_outcome as core_mgo
                from src.addons import model_explainability as add_shap
                """
            )
        ),
        nbf.v4.new_markdown_cell("## Recreate matchup dataset and train models"),
        nbf.v4.new_code_cell(
            dedent(
                """\
                team_season = core_dl.load_team_season()
                teams = core_dl.load_teams()
                team_features = core_fe.build_team_season_features(team_season, teams)
                matchups = core_fe.build_pairwise_matchups(team_features)

                train_df, test_df = core_mgo.season_train_test_split(matchups, train_ratio=0.7)
                X_train, y_train, feature_cols = core_mgo.prepare_features(train_df)
                X_test = test_df[feature_cols]
                y_test = test_df['label']

                models = core_mgo.train_models(train_df, feature_cols=feature_cols)
                rf_model = models['random_forest']
                print(f'Trained models: {list(models.keys())}')
                """
            )
        ),
        nbf.v4.new_markdown_cell("## Compute SHAP values for the random forest"),
        nbf.v4.new_code_cell(
            dedent(
                """\
                explainer, shap_values, X_sample = add_shap.compute_shap_values(rf_model, X_test)
                summary_path = FIG_DIR / 'shap_summary_rf.png'
                add_shap.plot_shap_summary(shap_values, X_sample, class_idx=1, save_path=summary_path)
                print(f'Saved SHAP summary plot to {summary_path}')

                dependence_path = FIG_DIR / 'shap_dependence_diff_win_pct.png'
                if 'diff_win_pct' in X_sample.columns:
                    add_shap.plot_shap_dependence(
                        shap_values,
                        X_sample,
                        feature_name='diff_win_pct',
                        class_idx=1,
                        save_path=dependence_path,
                    )
                    print(f'Saved SHAP dependence plot to {dependence_path}')
                else:
                    print('diff_win_pct not in feature set; skipping dependence plot.')
                """
            )
        ),
        nbf.v4.new_markdown_cell(
            "## Interpretations\nThe SHAP summary shows which matchup features consistently drive predictions (e.g., diff_win_pct, diff_margin). Dependence plots reveal how higher win-percentage gaps increase the likelihood of Team A winning, while turnover or pace differences play secondary roles."
        ),
        nbf.v4.new_markdown_cell(
            dedent(
                """\
                ## Summary: Should this Add-on be Included in the Final Report?
                SHAP improves transparency and is valuable when emphasising interpretability, but it increases runtime and requires additional explanation. Recommend presenting it as an optional interpretability appendix unless the rubric specifically rewards explainable AI techniques.
                """
            )
        ),
    ]

    write_notebook(nb_dir / "A2_model_explainability_shap.ipynb", nb2_cells)

    # Notebook A3
    nb3_cells = [
        nbf.v4.new_markdown_cell(
            dedent(
                """\
                # Project 2.0 ADD-ON: Monte Carlo Matchup Simulation

                *Experimental notebook distinct from the required coursework.* We extend the binary matchup model with Monte Carlo simulations to estimate win probabilities and expected score margins for hypothetical pairings.
                """
            )
        ),
        nbf.v4.new_markdown_cell(
            dedent(
                """\
                ## Advantages and Disadvantages of this Add-on
                - **Advantages:** Produces probabilistic forecasts, aligns with sports analytics practices, and supports scenario analysis for arbitrary matchups.
                - **Disadvantages:** Relies on distributional assumptions (Gaussian noise), adds computational cost, and introduces additional models that must be explained.
                """
            )
        ),
        nbf.v4.new_markdown_cell(
            "## How to run\n1. Install requirements (scikit-learn).\n2. Launch Jupyter from the project root.\n3. Execute cells to retrain models and run Monte Carlo simulations."
        ),
        nbf.v4.new_code_cell(
            fig_setup
            + dedent(
                """\
                from sklearn.linear_model import LinearRegression

                from src.core import data_loading as core_dl
                from src.core import feature_engineering as core_fe
                from src.core import models_game_outcome as core_mgo
                from src.addons import monte_carlo_matchups as add_mc
                """
            )
        ),
        nbf.v4.new_markdown_cell("## Prepare matchup data and train models"),
        nbf.v4.new_code_cell(
            dedent(
                """\
                team_season = core_dl.load_team_season()
                teams = core_dl.load_teams()
                team_features = core_fe.build_team_season_features(team_season, teams)
                matchups = core_fe.build_pairwise_matchups(team_features)

                train_df, test_df = core_mgo.season_train_test_split(matchups, train_ratio=0.7)
                X_train, y_train, feature_cols = core_mgo.prepare_features(train_df)
                X_test = test_df[feature_cols]
                y_test = test_df['label']

                models = core_mgo.train_models(train_df, feature_cols=feature_cols)
                rf_model = models['random_forest']
                print('Random forest ready for probabilistic simulation.')

                reg_features = [col for col in feature_cols if col != 'diff_margin']
                reg_model = LinearRegression()
                reg_model.fit(train_df[reg_features], train_df['diff_margin'])
                print('Fitted linear regression to approximate point differential using remaining features.')
                """
            )
        ),
        nbf.v4.new_markdown_cell("## Run simulations for sample matchups"),
        nbf.v4.new_code_cell(
            dedent(
                """\
                example_rows = test_df.sample(n=3, random_state=42)
                results = []
                for _, row in example_rows.iterrows():
                    x_cls = row[feature_cols].values
                    cls_summary = add_mc.simulate_matchup_from_classifier(rf_model, x_cls, n_sim=5000)

                    x_reg = row[reg_features].values
                    reg_summary = add_mc.simulate_matchup_from_regressor(reg_model, x_reg, n_sim=5000)

                    results.append({
                        'season': row['year'],
                        'team_a': row['team_name_a'],
                        'team_b': row['team_name_b'],
                        **{f'classifier_{k}': v for k, v in cls_summary.items()},
                        **{f'regressor_{k}': v for k, v in reg_summary.items()},
                    })

                results_df = pd.DataFrame(results)
                monte_table_path = TABLE_DIR / 'monte_carlo_example_matchups.csv'
                results_df.to_csv(monte_table_path, index=False)
                print(f'Saved Monte Carlo summary table to {monte_table_path}')
                results_df
                """
            )
        ),
        nbf.v4.new_markdown_cell("## Visualise simulated margin distribution"),
        nbf.v4.new_code_cell(
            dedent(
                """\
                first_row = example_rows.iloc[0]
                x_reg = first_row[reg_features].values
                reg_summary = add_mc.simulate_matchup_from_regressor(reg_model, x_reg, n_sim=10000, return_samples=True)
                draws = reg_summary.pop('samples')

                fig, ax = plt.subplots(figsize=(8, 4))
                sns.histplot(draws, bins=40, kde=True, ax=ax, color='#ff6b6b')
                ax.axvline(0, color='black', linestyle='--', label='Even matchup')
                ax.set_title(f\"Simulated margin distribution: {first_row['team_name_a']} vs {first_row['team_name_b']}\")
                ax.set_xlabel('Simulated margin (Team A - Team B)')
                ax.legend()
                fig.tight_layout()
                margin_fig_path = FIG_DIR / 'monte_carlo_margin_hist.png'
                fig.savefig(margin_fig_path, dpi=300)
                print(f'Saved margin histogram to {margin_fig_path}')
                fig
                """
            )
        ),
        nbf.v4.new_markdown_cell(
            dedent(
                """\
                ## Summary: Should this Add-on be Included in the Final Report?
                Monte Carlo simulations add storytelling value and mimic professional analytics, but they introduce new assumptions and require clear communication about noise models. Include this section as an exploratory appendix unless the rubric rewards probabilistic forecasting.
                """
            )
        ),
    ]

    write_notebook(nb_dir / "A3_monte_carlo_matchups.ipynb", nb3_cells)


if __name__ == "__main__":
    main()
