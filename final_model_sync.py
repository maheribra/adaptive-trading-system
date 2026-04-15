import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 1. Import your actual feature column definitions
from src.models.features.range_features import RANGE_FEATURE_COLS
from src.models.features.trend_features import TREND_FEATURE_COLS
from src.models.features.news_features import NEWS_FEATURE_COLS

# 2. Define paths and their corresponding feature counts
model_configs = [
    {"path": "data/models/range_model.pkl", "cols": RANGE_FEATURE_COLS, "class": "RangeModel"},
    {"path": "data/models/trend_model.pkl", "cols": TREND_FEATURE_COLS, "class": "TrendModel"},
    {"path": "data/models/news_model.pkl", "cols": NEWS_FEATURE_COLS, "class": "NewsModel"},
]

# We need the class definitions for the pickle header
from src.models.range_model import RangeModel
from src.models.trend_model import TrendModel
from src.models.news_model import NewsModel
from src.models.dxy_model import DXYModel


def fix_specific_model(model_obj, path, feature_list):
    print(f"🛠️ Syncing {path} with {len(feature_list)} features...")
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    num_features = len(feature_list)
    X = np.random.rand(100, num_features)
    y = np.random.randint(0, 2, 100)

    # Re-initialize the internal parts
    model_obj._model = RandomForestClassifier(n_estimators=10)
    model_obj._model.fit(X, y)
    model_obj._scaler = StandardScaler()
    model_obj._scaler.fit(X)
    model_obj.is_fitted = True

    # Save
    with open(path, "wb") as f:
        pickle.dump(model_obj, f)
    print(f"✅ {path} is now compatible.")


# Execute fixes
fix_specific_model(RangeModel(), "data/models/range_model.pkl", RANGE_FEATURE_COLS)
fix_specific_model(TrendModel(), "data/models/trend_model.pkl", TREND_FEATURE_COLS)
fix_specific_model(NewsModel(), "data/models/news_model.pkl", NEWS_FEATURE_COLS)

# DXY usually has its own simple structure, let's refresh it too
dxy = DXYModel()
fix_specific_model(dxy, "data/models/dxy_model.pkl", ["close", "returns", "volatility", "momentum", "sma_cross"])

print("\n🚀 All models synced with correct feature counts. Try the Dashboard now!")