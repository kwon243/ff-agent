# QB Model - Quick Start Guide

## For Your Team

### Key Files to Review

1. **README.md** - Full documentation
2. **model_performance_comparison.png** - Visual results
3. **qb_ensemble_predictions.csv** - All predictions
4. **FINAL_RESULTS.txt** - Complete results summary

### Production Model

**File**: `qb_ensemble_model.pkl`

```python
import pickle
with open('qb_ensemble_model.pkl', 'rb') as f:
    model = pickle.load(f)
```

### Results Summary

**Beat Baseline Model**: R² 0.1569 vs 0.1546 (+1.5%)  
**MAE**: 5.81 (essentially tied with 5.79)  
**Top Feature**: Rolling averages (last_5_games_ppr, season_avg_ppr)  
**New Features**: 41% of model importance

### File Structure

```
qb_model/
├── README.md                    # Main documentation
├── QUICKSTART.md               # This file
├── requirements.txt            # Python dependencies
│
├── Models (Production)
│   ├── qb_ensemble_model.pkl   # Main model
│   ├── qb_xgb_final.pkl
│   ├── qb_lgb_final.pkl
│   └── qb_cat_final.pkl
│
├── Results
│   ├── qb_ensemble_predictions.csv
│   ├── model_performance_comparison.png
│   └── qb_final_features.txt
│
├── Code
│   ├── qb_full_pipeline.py     # Training pipeline
│   └── qb_feature_engineering.py
│
└── Documentation
    ├── FINAL_RESULTS.txt
    ├── DATA_INSIGHTS.md
    ├── FEATURE_ENGINEERING_SUMMARY.md
    └── PIPELINE_OVERVIEW.md
```

### Install & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Retrain model (optional)
python qb_full_pipeline.py

# Takes ~20 minutes
```

---

**Author**: Minjun  
**Date**: November 2025  
**Status**: Production Ready
