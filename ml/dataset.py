from __future__ import annotations
import os
import pandas as pd
from typing import Tuple

from .config import ASSIGNMENT_MCAP, ASSIGNMENT_SECTOR, ASSIGNMENT_RULES, RANK_THRESHOLD


def load_raw() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	mcap = pd.read_csv(ASSIGNMENT_MCAP)
	sector = pd.read_csv(ASSIGNMENT_SECTOR)
	rules = pd.read_csv(ASSIGNMENT_RULES)
	return mcap, sector, rules


def melt_mcap(ranked_mcap: pd.DataFrame) -> pd.DataFrame:
	wide = ranked_mcap.copy()
	wide.columns = ["CO_NAME"] + list(wide.columns[1:])
	long = wide.melt(id_vars=["CO_NAME"], var_name="year", value_name="rank")
	long["year"] = pd.to_numeric(long["year"], errors="coerce")
	long = long.dropna(subset=["year"]).astype({"year": int})
	return long


def build_dataset() -> pd.DataFrame:
	mcap, sector, rules = load_raw()
	# Sector normalize key
	sector_kv = sector.copy()
	sector_kv["key"] = sector_kv["CO_NAME"].str.strip().str.lower()
	sector_kv = sector_kv[["key", "Sector"]]

	# MCAP long with normalized key
	mcap_long = melt_mcap(mcap)
	mcap_long["key"] = mcap_long["CO_NAME"].str.strip().str.lower()
	mcap_long = mcap_long.merge(sector_kv, on="key", how="left")

	# Engineer coarse features: year, rank, sector one-hot later; add rank bins
	mcap_long["is_top_50"] = (mcap_long["rank"] <= 50).astype(int)
	mcap_long["is_top_100"] = (mcap_long["rank"] <= 100).astype(int)
	mcap_long["is_top_500"] = (mcap_long["rank"] <= RANK_THRESHOLD).astype(int)

	# Proxy target: whether stock is in top 500 in next year
	next_year = mcap_long[["key", "year", "rank"]].copy()
	next_year["year"] = next_year["year"] - 1
	next_year = next_year.rename(columns={"rank": "next_rank"})

	df = mcap_long.merge(next_year, on=["key", "year"], how="left")
	df["label"] = (df["next_rank"] <= RANK_THRESHOLD).fillna(0).astype(int)

	# Select features and target
	return df[[
		"CO_NAME", "key", "year", "rank", "Sector",
		"is_top_50", "is_top_100", "is_top_500", "label"
	]].dropna(subset=["rank"])
