import glob
import os
import numpy as np
import pandas as pd

# Use an absolute path or relative path mapping to the exact workspace root
# Let's search using the full absolute path from your terminal working directory
runs_dir = "runs/segment/Surgical_Instrument_Segmentation"

# Find all fold folders matching the pattern
fold_dirs = sorted(glob.glob(os.path.join(runs_dir, "l50_folded_*")))

if not fold_dirs:
  print(f"No fold directories found in '{runs_dir}'. Please check your path.")
else:
  metrics_summary = []
  print(
      f"Found {len(fold_dirs)} folds in '{runs_dir}'. Processing"
      f" results...\n"
  )

  for fold_path in fold_dirs:
    fold_name = os.path.basename(fold_path)
    csv_file = os.path.join(fold_path, "results.csv")

    if not os.path.exists(csv_file):
      print(f"Warning: {csv_file} not found. Skipping {fold_name}.")
      continue

    # Load CSV and strip whitespace from column names
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()

    # Find the best epoch based on Mask mAP50-95
    best_row = df.loc[df["metrics/mAP50-95(M)"].idxmax()]
    best_epoch = int(best_row["epoch"])

    # Extract relevant metrics for this fold's best epoch
    fold_metrics = {
        "Fold": fold_name,
        "Best Epoch": best_epoch,
        "Precision (M)": best_row["metrics/precision(M)"],
        "Recall (M)": best_row["metrics/recall(M)"],
        "mAP50 (M)": best_row["metrics/mAP50(M)"],
        "mAP50-95 (M)": best_row["metrics/mAP50-95(M)"],
        "Precision (B)": best_row["metrics/precision(B)"],
        "Recall (B)": best_row["metrics/recall(B)"],
        "mAP50 (B)": best_row["metrics/mAP50(B)"],
        "mAP50-95 (B)": best_row["metrics/mAP50-95(B)"],
    }
    metrics_summary.append(fold_metrics)

  # Convert summary to DataFrame
  summary_df = pd.DataFrame(metrics_summary)

  print("=== INDIVIDUAL FOLD BEST RESULTS ===")
  print(
      summary_df.to_string(
          index=False,
          formatters={
              "Precision (M)": "{:.4f}".format,
              "Recall (M)": "{:.4f}".format,
              "mAP50 (M)": "{:.4f}".format,
              "mAP50-95 (M)": "{:.4f}".format,
              "Precision (B)": "{:.4f}".format,
              "Recall (B)": "{:.4f}".format,
              "mAP50 (B)": "{:.4f}".format,
              "mAP50-95 (B)": "{:.4f}".format,
          },
      )
  )
  print("\n" + "=" * 50 + "\n")

  # Calculate Mean and Standard Deviation across all folds
  numeric_cols = [
      "Precision (M)",
      "Recall (M)",
      "mAP50 (M)",
      "mAP50-95 (M)",
      "Precision (B)",
      "Recall (B)",
      "mAP50 (B)",
      "mAP50-95 (B)",
  ]

  mean_vals = summary_df[numeric_cols].mean()
  std_vals = summary_df[numeric_cols].std()

  final_report = pd.DataFrame({"Mean": mean_vals, "Std Dev": std_vals}).T

  print("=== FINAL CROSS-VALIDATION REPORT (Mean ± Std) ===")
  print(final_report.to_string())