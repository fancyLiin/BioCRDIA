#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json


# =====================================================
# 1. Configuration
# =====================================================

INPUT_FILES = {
    "train": "../train_data_taxsft/biocrdia_tax_train.json",
    "val": "../train_data_taxsft/biocrdia_tax_val.json",
    "test": "../train_data_taxsft/biocrdia_tax_test.json"
}

OUTPUT_FILES = {
    "train": "../train_data_taxsft/biocrdia_vanilla_train.json",
    "val": "../train_data_taxsft/biocrdia_vanilla_val.json",
    "test": "../train_data_taxsft/biocrdia_vanilla_test.json"
}

VANILLA_INSTRUCTION = (
    "You are BioCRDIA, a process-oriented bioinformatics intelligent tutoring system. "
    "Provide pedagogical feedback based on the student's execution state."
)


# =====================================================
# 2. Output conversion
# =====================================================

def strip_diagnostic_label(output_text):
    """
    Remove:
    ### 0. Diagnostic Label
    Error Type: ...
    Evidence: ...

    Keep:
    ### 1. Defect Analysis
    ### 2. Repair Logic
    ### 3. Pedagogical Scaffolding
    """

    marker = "### 1. Defect Analysis"

    if marker in output_text:
        return output_text[output_text.find(marker):].strip()

    # fallback: if the expected marker is missing, keep original text
    # but this should rarely happen if your TaxSFT data is clean
    return output_text.strip()


def convert_item(item):
    vanilla_output = strip_diagnostic_label(item["output"])

    return {
        "id": item["id"],
        "instruction": VANILLA_INSTRUCTION,
        "input": item["input"],
        "output": vanilla_output,

        # metadata，不参与训练，但后续评价要用
        "domain": item.get("domain", ""),
        "error_type": item.get("error_type", ""),
        "sub_error_type": item.get("sub_error_type", ""),
        "reference_solution": item.get("reference_solution", "")
    }


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
    print("Creating Vanilla SFT dataset from TaxSFT dataset")
    print("=" * 80)

    for split in ["train", "val", "test"]:
        input_path = INPUT_FILES[split]
        output_path = OUTPUT_FILES[split]

        data = load_json(input_path)

        converted = [convert_item(item) for item in data]

        save_json(converted, output_path)

        print(f"{split}: {len(converted)}")
        print(f"Saved to: {output_path}")

        # quick sanity check
        sample_output = converted[0]["output"]

        if "### 0. Diagnostic Label" in sample_output:
            print(f"[Warning] Diagnostic Label still exists in {split} output.")

        if "### 1. Defect Analysis" not in sample_output:
            print(f"[Warning] Defect Analysis marker missing in {split} output.")

    print("=" * 80)
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()