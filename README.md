# CS441 Fantasy Football Agent

A machine learning pipeline that predicts weekly PPR fantasy football points for QBs, RBs, WRs, and TEs using player performance grades, team defensive metrics, and draft ranking data.

## How It Works

The pipeline has four stages:

```
1. Scrape  →  2. Preprocess  →  3. Merge  →  4. Predict
```

**Scrape**: Pull raw data from PFF (player grades), StatHead (weekly fantasy points), and FantasyPros (draft rankings) using Selenium.

**Preprocess**: Clean and normalize each data source into standardized CSVs keyed by `PFR_ID` (Pro Football Reference player ID).

**Merge**: Join player grades, weekly fantasy points, and draft ranking data into a single feature dataset per position.

**Predict**: Load pre-trained Kernel Ridge regression models and generate weekly PPR point predictions for the 2024 season.

## Requirements

```bash
pip install -r requirements.txt
```

You also need:
- Chrome browser (for Selenium scrapers)
- ChromeDriver matching your Chrome version
- Pre-trained model files in `modelling_trials/avery/`:
  - `qb_kernel_ridge.pkl`
  - `rb_kernel_ridge.pkl`
  - `wr_te_kernel_ridge.pkl`

## Usage

### Step 1 — Data Collection (optional if CSVs already exist)

The scrapers require a running Chrome instance and manual login to premium data sources.

```bash
# Scrape PFF player grades (requires PFF premium account, manual login)
python PFR.py

# Scrape StatHead weekly fantasy points (requires Chrome in debug mode)
python fetch_weekly_points.py
```

> These scrapers write raw CSVs to `new_qb_grades/`, `new_weekly_fp/`, and `new_draft_rankings/`.

### Step 2 — Preprocess

```bash
python qb_grades_preprocessing.py
python rb_grades_preprocessing.py
python wr_and_te_grades_preprocessing.py
python draft_rank_preprocessing.py
python weekly_fp_preprocessing.py
```

### Step 3 — Merge

```bash
python qb_merge.py          # → final_qb_data.csv
python rb_merge.py          # → final_rb_data.csv
python wr_and_te_merge.py   # → final_wr_and_te_data.csv
```

Optionally validate output:

```bash
python data_checker.py
```

### Step 4 — Predict

Open and run the main notebook:

```bash
jupyter notebook competitive_gymnasium.ipynb
```

The notebook loads `gymnasium_*_data_2024.csv` files, applies the three trained models, and outputs weekly predictions (e.g., `qb_ppr_predictions_by_week.csv`).

## File Overview

| File | Purpose |
|---|---|
| `PFR.py` | Selenium scraper for PFF player grades |
| `PFRHelper.py` | Player name/ID matching utility across data sources |
| `fetch_weekly_points.py` | Selenium scraper for StatHead weekly fantasy points |
| `qb_grades_preprocessing.py` | Clean and normalize raw QB grade CSVs |
| `rb_grades_preprocessing.py` | Clean and normalize raw RB grade CSVs |
| `wr_and_te_grades_preprocessing.py` | Clean and normalize raw WR/TE grade CSVs |
| `draft_rank_preprocessing.py` | Clean FantasyPros draft ranking data |
| `weekly_fp_preprocessing.py` | Aggregate weekly fantasy points with team defensive stats |
| `qb_merge.py` | Join QB grades + weekly FP + draft rankings |
| `rb_merge.py` | Join RB grades + weekly FP + draft rankings |
| `wr_and_te_merge.py` | Join WR/TE grades + weekly FP + draft rankings |
| `gymnasium_models.py` | Kernel Ridge model wrappers (QB, RB, WR/TE) |
| `competitive_gymnasium.ipynb` | Main notebook — loads 2024 data and runs predictions |
| `data_checker.py` | Validates merged CSV integrity |

## Data Sources

- **PFF (Pro Football Focus)** — Player grades (2011–2024), premium subscription required
- **StatHead** — Weekly PPR fantasy points (2012–2024)
- **FantasyPros** — Draft rankings / ECR (2012–2024)

Player records are matched across sources using `PFR_ID` (Pro Football Reference player ID). The `PFRHelper.py` utility handles name normalization for players with initials, compound names, or accented characters.

## Models

Three separate Kernel Ridge regression models, one per position group:

| Model | Key features |
|---|---|
| QB | Scramble rate, rushing, opponent pass rush/coverage/defense grades, home/away |
| RB | Receiving grade, blocking grades, opponent run defense, ECR-ADP gap |
| WR/TE | Target share, YAC, route running, opponent coverage grade |

All models are pre-trained and stored as pickle files. Bye weeks always receive a prediction of `0.0`.

## Notes

- All scripts use paths relative to the repo root — run them from the repo root directory.
- The scraping scripts were written against specific site layouts and may need updates if those sites change their HTML structure.
- This was a course project (CS441); the `modelling_trials/` directory contains experimental model files that are not tracked in git.
