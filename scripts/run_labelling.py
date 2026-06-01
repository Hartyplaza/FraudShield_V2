"""
Step 1b: Generate fraud labels on parsed logs.
Output: data/processed/labelled_logs.csv

Run after run_ingestion.py:
    python scripts/run_ingestion.py
    python scripts/run_labelling.py
"""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, ".")

import yaml
from src.ingestion.fraud_simulator import compute_fraud_scores, summarise_fraud_signals

cfg      = yaml.safe_load(open("configs/pipeline.yaml"))
proc_dir = Path(cfg["data"]["processed_path"])

# Load parsed logs
parsed_path = proc_dir / "parsed_logs.csv"
print(f"Loading parsed logs from {parsed_path} ...")
df = pd.read_csv(parsed_path, parse_dates=["timestamp"])
print(f"Input shape: {df.shape}")

# Generate labels
df = compute_fraud_scores(df, random_seed=cfg["project"]["random_seed"])

# Signal quality report
print("\nSignal breakdown (fraud vs legitimate):")
summary = summarise_fraud_signals(df)
print(summary.to_string(index=False))

# Save
out = proc_dir / "labelled_logs.csv"
df.to_csv(out, index=False)
print(f"\nLabelled dataset saved → {out}")
print(f"Columns: {df.columns.tolist()}")
