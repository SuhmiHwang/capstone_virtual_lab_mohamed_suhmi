"""Calculate the 3D shape descriptors for many polymers and save them to a CSV file.

RDKit can crash or hang on a few unusual structures while generating a conformer.
Each worker therefore runs in its own process and writes every result immediately;
if a worker crashes or needs longer than the timeout for one polymer, that polymer
gets NaN and the worker is restarted after it. An interrupted run resumes from the
results already written.

Usage (from the project root):
    python -m src.shape_cache data/raw/train.csv data/processed/shape_descriptors.csv
"""

import csv
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.polymer_features import SHAPE_COLUMNS, calculate_shape_descriptors

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _work(smiles_file, out_file, start, stop):
    """Worker: process rows start..stop-1, appending one line per polymer."""
    smiles = pd.read_csv(smiles_file)["SMILES"]
    with open(out_file, "a", newline="") as handle:
        writer = csv.writer(handle)
        for idx in range(start, stop):
            values = calculate_shape_descriptors(smiles[idx])
            writer.writerow([idx] + [values[col] for col in SHAPE_COLUMNS])
            handle.flush()


def compute_shape_cache(smiles_file, out_file, n_workers=8, timeout=120):
    """Driver: split the rows over several workers and restart any that crash or hang."""
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    n_rows = len(pd.read_csv(smiles_file))
    bounds = np.linspace(0, n_rows, n_workers + 1).astype(int)
    # Existing part files are kept, so an interrupted calculation resumes where it stopped
    parts = [out_file.with_suffix(f".part{k}.csv") for k in range(n_workers)]

    def launch(k, start):
        return subprocess.Popen(
            [sys.executable, "-m", "src.shape_cache", "--work", str(smiles_file), str(parts[k]),
             str(start), str(bounds[k + 1])],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL,
        )

    def next_row(k):
        if not parts[k].exists():
            return bounds[k]
        done = [int(line.split(",")[0]) for line in parts[k].read_text().splitlines() if line]
        return max(done) + 1 if done else bounds[k]

    running, started = {}, {}
    for k in range(n_workers):
        if next_row(k) < bounds[k + 1]:
            running[k], started[k] = launch(k, next_row(k)), (time.time(), next_row(k))

    skipped = []
    while running:
        time.sleep(2)
        for k, process in list(running.items()):
            row = next_row(k)
            if process.poll() is None:
                # Progress check: a polymer that takes longer than the timeout is skipped
                if row != started[k][1]:
                    started[k] = (time.time(), row)
                    continue
                if time.time() - started[k][0] < timeout:
                    continue
                process.kill()
                process.wait()
            elif process.returncode == 0 or row >= bounds[k + 1]:
                del running[k]
                continue

            # The worker crashed or timed out on this row: record NaN and continue after it
            skipped.append(row)
            with open(parts[k], "a", newline="") as handle:
                csv.writer(handle).writerow([row] + [""] * len(SHAPE_COLUMNS))
            if row + 1 < bounds[k + 1]:
                running[k], started[k] = launch(k, row + 1), (time.time(), row + 1)
            else:
                del running[k]

    shape_df = pd.concat(
        [pd.read_csv(part, header=None, names=["index"] + SHAPE_COLUMNS) for part in parts]
    ).set_index("index").sort_index()
    shape_df.index.name = None
    shape_df = shape_df.reindex(range(n_rows))
    shape_df.to_csv(out_file)
    for part in parts:
        part.unlink()
    return shape_df, skipped


if __name__ == "__main__":
    if sys.argv[1] == "--work":
        _work(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
    else:
        result, skipped_rows = compute_shape_cache(sys.argv[1], sys.argv[2])
        print(f"{len(result)} rows, {len(skipped_rows)} skipped: {skipped_rows}")
