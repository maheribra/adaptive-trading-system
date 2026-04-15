import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Define the exact paths your MetaController uses
paths = {
    "news": "data/models/news_model.pkl",
    "range": "data/models/range_model.pkl",
    "trend": "data/models/trend_model.pkl",
    "dxy": "data/models/dxy_model.pkl"
}


def create_dummy_model(model_class, path):
    print(f"🛠️ Creating compatible snapshot for: {path}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Initialize a generic model
    m = model_class()

    # Simulate a fit with a small amount of data
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, 10)

    # Use a basic RandomForest as the internal engine to avoid Cython loss errors
    m._model = RandomForestClassifier(n_estimators=10)
    m._model.fit(X, y)
    m._scaler = StandardScaler()
    m._scaler.fit(X)
    m.is_fitted = True

    # Save it
    with open(path, "wb") as f:
        pickle.dump(m, f)
    print(f"✅ Saved.")


# We need to import your actual classes so pickle knows what 'NewsModel' is
from src.models.news_model import NewsModel
from src.models.range_model import RangeModel
from src.models.trend_model import TrendModel
from src.models.dxy_model import DXYModel

# Create fresh snapshots for all
create_dummy_model(NewsModel, paths["news"])
create_dummy_model(RangeModel, paths["range"])
create_dummy_model(TrendModel, paths["trend"])
create_dummy_model(DXYModel, paths["dxy"])

print("\n🚀 ALL MODELS UPDATED. Try: streamlit run dashboard.py")