from __future__ import annotations
import os
import joblib
import pandas as pd
from typing import Tuple

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from .config import MODEL_PATH, PREPROCESS_PATH
from .dataset import build_dataset


def train_model(force_retrain: bool = False) -> Tuple[float, str]:
	if os.path.exists(MODEL_PATH) and not force_retrain:
		return 0.0, "model_exists"

	df = build_dataset()

	feature_columns_numeric = ["year", "rank", "is_top_50", "is_top_100", "is_top_500"]
	feature_columns_categorical = ["Sector"]
	target_column = "label"

	X = df[feature_columns_numeric + feature_columns_categorical]
	y = df[target_column]

	X_train, X_valid, y_train, y_valid = train_test_split(
		X, y, test_size=0.2, random_state=42, stratify=y
	)

	preprocess = ColumnTransformer(
		transformers=[
			("num", StandardScaler(with_mean=False), feature_columns_numeric),
			("cat", OneHotEncoder(handle_unknown="ignore"), feature_columns_categorical),
		]
	)

	clf = LogisticRegression(max_iter=200, class_weight="balanced")
	pipe = Pipeline(steps=[("prep", preprocess), ("clf", clf)])
	pipe.fit(X_train, y_train)

	proba = pipe.predict_proba(X_valid)[:, 1]
	auc = roc_auc_score(y_valid, proba)

	joblib.dump(pipe, MODEL_PATH)
	joblib.dump({
		"features_numeric": feature_columns_numeric,
		"features_categorical": feature_columns_categorical,
		"target": target_column,
	}, PREPROCESS_PATH)

	return auc, MODEL_PATH


def ensure_model(force_retrain: bool = False) -> None:
	if not os.path.exists(MODEL_PATH) or force_retrain:
		train_model(force_retrain=True)
