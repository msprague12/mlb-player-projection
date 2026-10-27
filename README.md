# MLB Player Performance Projection

A machine learning system that projects MLB hitters' batting statistics up to several seasons ahead, using Random Forest and XGBoost models trained on 11 seasons of FanGraphs data (2013-2023).

Built for CS 549 (Machine Learning) at San Diego State University. Full write-up: [`report/CS_549_Final_Project_Report.pdf`](report/CS_549_Final_Project_Report.pdf).

## What it does

Given a player's name and a starting season, the tool:

1. Looks up the player's season stats.
2. Predicts the next season's stats with both a Random Forest and an XGBoost model.
3. Feeds the averaged predictions back in as features to project further seasons (a "cascading" projection).
4. Prints a table of projections and plots any stat you choose.

**Targets (7):** AVG, OBP, SLG, HR, RBI, wOBA, wRC+
**Features (15):** Age, G, PA, AVG, OBP, SLG, HR, RBI, BB%, K%, ISO, BABIP, wOBA, wRC+, WAR

That is 14 models in total (7 stats x 2 algorithms). Training pairs are built from consecutive seasons: a player's stats in season *t* are the features, and their stats in season *t+1* are the targets. Player-seasons with fewer than 100 plate appearances are excluded.

## Results

Training-set mean squared error (see Limitations below):

| Stat | Random Forest | XGBoost |
|------|---------------|---------|
| AVG  | 0.0001 | 0.0001 |
| OBP  | 0.0002 | 0.0001 |
| SLG  | 0.0007 | 0.0004 |
| wOBA | 0.0002 | 0.0001 |
| HR   | 9.5712 | 4.0798 |
| RBI  | 75.8823 | 38.3406 |
| wRC+ | 92.0807 | 46.4729 |

- XGBoost had lower training error than Random Forest on all 7 targets.
- Rate statistics (AVG, OBP, SLG, wOBA) show very small error. Counting stats (HR, RBI) and wRC+ show much larger error, which fits their higher year-to-year volatility and dependence on playing time.
- Multi-year projections were checked qualitatively against known aging-curve patterns using Mike Trout (veteran, projected decline in his late 30s) and Bobby Witt Jr. (young star, projected to hold peak production). See the report for the plots.

## Limitations

I want to be upfront about what this project does and does not show:

- **No held-out evaluation.** The MSE values above are computed on the same data the models were trained on, so they measure fit, not predictive accuracy. Training error for tree ensembles is optimistic, particularly for Random Forest.
- **No baseline comparison.** The models were not compared against simple baselines such as "next year equals this year" or the Marcel projection system.
- **Cascading error.** Multi-year projections reuse earlier predictions as inputs, so errors compound the further out you project, and projections tend to drift toward the league average.
- **Validation of multi-year output is qualitative only.**
- **Limited features.** No Statcast data (exit velocity, launch angle), injury history, or minor league data.

Natural next steps: a train/test split by season, comparison to a Marcel-style baseline, and per-year error analysis.

## Data access note

Data was pulled with [`pybaseball`](https://github.com/jldbc/pybaseball) from FanGraphs (`batting_stats`) when the project was built in 2025. FanGraphs has since started blocking automated requests (HTTP 403) and has put data exports behind a membership, so **the live data pull no longer works** and the tool cannot be run end to end as written. The results in the report are from the original runs.

## Setup (for reference)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/StatPredictor.py
```

On first run the script trains and saves `projection_models.pkl`, then reuses it. Run it from the repository root so the file is found. The saved models are not committed because pickled scikit-learn models can break across library versions.

## Repository structure

```
.
├── src/StatPredictor.py     # data collection, training, projection, plotting (CLI)
├── report/                  # final report (PDF)
├── requirements.txt
└── README.md
```

## Tech

Python, pandas, NumPy, scikit-learn (Random Forest), XGBoost, Matplotlib, pybaseball.

## Author

Matthew Sprague, B.S. Computer Science, San Diego State University
[LinkedIn](https://www.linkedin.com/in/matthew-sprague-478b55325)
