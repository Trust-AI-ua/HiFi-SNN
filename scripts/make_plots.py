#!/usr/bin/env python3
"""Generate the Phase-1 SNN-sweep result figures and tables.

Reads results/summary.csv + results/<dataset>/SNN_*/<encoding>/results.json
and writes figures (PDF + PNG) + a ranked CSV under results/<dataset>/figures/.

Usage:
    python scripts/make_plots.py nslkdd
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.plotting import generate_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="dataset name (e.g. nslkdd)")
    args = ap.parse_args()

    out_dir = REPO_ROOT / "results" / args.dataset / "figures"
    generate_all(args.dataset, REPO_ROOT, out_dir)


if __name__ == "__main__":
    main()
