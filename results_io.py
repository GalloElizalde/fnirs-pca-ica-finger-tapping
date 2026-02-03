import os
import csv

# Auxiliar function to save csv with the analysis results

# Define columns
FIELDNAMES = ["subject", "chrom", "method", "recon_error", "best_abs_corr", "best_component", "best_corr"]


def append_row(csv_path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            w.writeheader()
        w.writerow(row)
