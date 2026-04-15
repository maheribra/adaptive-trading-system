import pandas as pd
import numpy as np
import pickle
from src.models.news_model import NewsModel
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# 1. Configuration
SAVE_PATH = str(Path("data/models/news_model.pkl").absolute())

print("🛠️ Creating a dummy compatible NewsModel...")

# Initialize the model
model = NewsModel()

# Manually simulate a 'fitted' state so we can save it
# We use a tiny bit of random data just to satisfy scikit-learn's structure
X_dummy = np.random.rand(100, 15) # 15 is the number of NEWS_FEATURE_COLS
y_dummy = np.random.randint(0, 2, 100)

model._model = GradientBoostingClassifier(**model.params)
model._model.fit(X_dummy, y_dummy)
model._scaler = StandardScaler()
model._scaler.fit(X_dummy)
model.is_fitted = True

# Define the feature importance series to match expected NEWS_FEATURE_COLS
# These are the columns your script was complaining about
feature_names = [
    'spike_magnitude', 'spike_direction', 'bars_since_spike', 'post_spike_drift',
    'pre_spike_vol', 'body_ratio', 'upper_wick_ratio', 'lower_wick_ratio',
    'close_position', 'momentum_3', 'momentum_10', 'vol_acceleration',
    'range_expansion', 'candle_direction', 'mean_reversion_pressure'
]
model.feature_importance_ = pd.Series(
    model._model.feature_importances_,
    index=feature_names
)

print(f"💾 Saving compatible pickle to: {SAVE_PATH}")
model.save(SAVE_PATH)

print("\n✅ Success! The 'Pickle Ghost' is gone.")
print("🚀 Now run: streamlit run dashboard.py")