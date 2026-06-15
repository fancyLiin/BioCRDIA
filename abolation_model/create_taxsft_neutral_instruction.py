#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os


# =====================================================
# 1. Configuration
# =====================================================

INPUT_FILES = {
    "train": "../train_data_taxsft/biocrdia_tax_train.json",
    "val": "../train_data_taxsft/biocrdia_tax_val.json",
    "test": "../train_data_taxsft/biocrdia_tax_test.json"
}

OUTPUT_FILES = {
    "train": "train_data/biocrdia_tax_neutral_train.json",
    "val": "train_data/biocrdia_tax_neutral_val.json",
    "test": "train_data/biocrdia_tax_neutral_test.json"
}

NEUTRAL_INSTRUCTION = (
    "You are BioCRDIA, a process-oriented bioinformatics intelligent tutoring system. "
    "Provide pedagogical feedback based on the student's execution state."
)


# =====================================================
# 2. IO
# =====================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =====================================================
# 3. Main
# =====================================================

def main():
    print("=" * 80)
    print("Creating TaxSFT dataset with neutral instruction")
    print("=" * 80)

    for split in ["train", "val", "test"]:
        input_path = INPUT_FILES[split]
        output_path = OUTPUT_FILES[split]

        data = load_json(input_path)

        for item in data:
            item["instruction"] = NEUTRAL_INSTRUCTION

        save_json(data, output_path)

        print(f"{split}: {len(data)} -> {output_path}")

        # sanity check
        print("Sample instruction:")
        print(data[0]["instruction"])
        print("Sample output starts with:")
        print(data[0]["output"][:120])
        print("-" * 80)

    print("Done.")


if __name__ == "__main__":
    main()