
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pawpularity_project.data import load_csv, save_csv
from pawpularity_project.model import load_model

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/model.joblib")
    ap.add_argument("--input", required=True, help="CSV with feature columns")
    ap.add_argument("--output", default="predictions.csv")
    ap.add_argument("--id-col", default="id", help="ID column for submission output")
    return ap.parse_args()


def main():
    args = parse_args()

    df = load_csv(args.input)
    ids = df[args.id_col]

    model = load_model(args.model)

    preds = model.predict(df).reshape(-1)

    submission = pd.DataFrame(
        {
            "Id": ids,
            "Pawpularity": preds.clip(0, 100),
        }
    )

    save_csv(out, args.output)
    print(f"Saved predictions to: {args.output}")


if __name__ == "__main__":
    main()
