from __future__ import annotations
import os
import joblib
import pandas as pd
from typing import Dict, Any

from .config import MODEL_PATH, PREPROCESS_PATH


def _load_model():
	if not os.path.exists(MODEL_PATH):
		raise RuntimeError("Model not trained yet.")
	return joblib.load(MODEL_PATH)


def _load_meta():
	return joblib.load(PREPROCESS_PATH)


def predict_from_features(payload: Dict[str, Any]) -> Dict[str, Any]:
	meta = _load_meta()
	pipe = _load_model()

	# Expected features
	num = meta["features_numeric"]
	cat = meta["features_categorical"]
	expected_order = num + cat

	# Default scaffold
	features: Dict[str, Any] = {k: None for k in expected_order}
	features.update(payload or {})

	# Type conversions and simple derivations
	if features.get("year") is not None:
		features["year"] = int(features["year"])
	if features.get("rank") is not None:
		features["rank"] = float(features["rank"])
	# Derive boolean flags if not provided
	if features.get("rank") is not None:
		features.setdefault("is_top_50", int(features["rank"] <= 50))
		features.setdefault("is_top_100", int(features["rank"] <= 100))
		features.setdefault("is_top_500", int(features["rank"] <= 500))
	# Normalize provided boolean-like values
	for b in ["is_top_50", "is_top_100", "is_top_500"]:
		if features.get(b) is not None:
			features[b] = int(features[b])
	# Ensure categorical present (can be None; OneHotEncoder handle_unknown="ignore")
	if "Sector" in features and features["Sector"] is not None:
		features["Sector"] = str(features["Sector"]) 

	# Build DataFrame with named columns to satisfy ColumnTransformer
	row = {k: features.get(k) for k in expected_order}
	X = pd.DataFrame([row], columns=expected_order)

	proba = float(pipe.predict_proba(X)[0][1])
	return {"prob_selected_next_year": proba}
