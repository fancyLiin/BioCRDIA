#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
from collections import defaultdict, Counter


# =====================================================
# 1. Configuration
# =====================================================

JUDGE_FILE = "./result_data/judge_feedback_quality_deepseek_zeroshot_vanilla_taxsft.json"

MODEL_NAMES = [
    "Zero-shot",
    "Vanilla SFT",
    "TaxSFT"
]

METRICS = [
    "diagnosis_accuracy",
    "repair_correctness",
    "bioinformatics_relevance",
    "pedagogical_quality"
]


# =====================================================
# 2. Utility
# =====================================================

def mean(values):
    return sum(values) / len(values) if values else 0.0


def std(values):
    if len(values) <= 1:
        return 0.0

    m = mean(values)
    return math.sqrt(
        sum((x - m) ** 2 for x in values) / (len(values) - 1)
    )


def ci95(values):
    if len(values) <= 1:
        return 0.0

    return 1.96 * std(values) / math.sqrt(len(values))


# =====================================================
# 3. Main
# =====================================================

def main():

    with open(JUDGE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = [x for x in data if x.get("status") == "success"]

    print("=" * 80)
    print("LLM-as-a-Judge Feedback Quality Summary")
    print("=" * 80)
    print(f"Valid judged samples: {len(data)}")

    scores_by_model = {
        model: {metric: [] for metric in METRICS}
        for model in MODEL_NAMES
    }

    best_counter = Counter()

    for item in data:
        mapping = item["mapping"]
        scores = item["scores"]

        best_model = item.get("best_model", "tie")
        best_counter[best_model] += 1

        for side in ["A", "B", "C"]:
            model_name = mapping[side]

            if model_name not in scores_by_model:
                continue

            for metric in METRICS:
                key = f"{metric}_{side}"
                value = scores[key]
                scores_by_model[model_name][metric].append(value)

    print("\n" + "=" * 80)
    print("Average Scores by Model")
    print("=" * 80)

    summary = {}

    for model in MODEL_NAMES:
        print(f"\n{model}")

        summary[model] = {}

        metric_means = []

        for metric in METRICS:
            values = scores_by_model[model][metric]
            m = mean(values)
            c = ci95(values)

            metric_means.append(m)

            summary[model][metric] = {
                "mean": m,
                "ci95": c,
                "n": len(values)
            }

            print(f"{metric}: {m:.3f} ± {c:.3f} (95% CI)")

        overall = mean(metric_means)

        summary[model]["overall"] = {
            "mean": overall
        }

        print(f"overall: {overall:.3f}")

    print("\n" + "=" * 80)
    print("Best-response Count")
    print("=" * 80)

    total = len(data)

    for model in MODEL_NAMES + ["tie"]:
        count = best_counter[model]
        ratio = count / total if total else 0.0
        print(f"{model}: {count}/{total} = {ratio:.2%}")

    summary["_best_response_count"] = dict(best_counter)
    summary["_valid_samples"] = total

    output_file = JUDGE_FILE.replace(".json", "_summary.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f"Summary JSON saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()