"""Create small, balanced EMBER train and test CSV datasets."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ember import prepare_ember_csvs


def run(ember_dir: str, output_dir: str, samples_per_class: int, max_records: int) -> tuple[str, str]:
    """Prepare labeled CSVs only; does not train models or modify the app."""
    train_csv, test_csv = prepare_ember_csvs(
        ember_dir, output_dir, samples_per_class, max_records
    )
    print(f"EMBER train CSV: {train_csv}")
    print(f"EMBER test CSV:  {test_csv}")
    print(f"Samples per class: {samples_per_class}")
    print("No models or dashboard artifacts were created.")
    return train_csv, test_csv


def main() -> None:
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ember-dir", default=os.path.join(project_dir, "ember"))
    parser.add_argument("--output-dir", default=os.path.join(project_dir, "data", "ember"))
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--max-records", type=int, default=250_000)
    args = parser.parse_args()
    run(args.ember_dir, args.output_dir, args.samples_per_class, args.max_records)


if __name__ == "__main__":
    main()