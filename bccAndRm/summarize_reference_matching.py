#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
from collections import Counter


# =====================================================
# 1. Configuration
# =====================================================

JUDGE_FILE = "./result_data/judge_reference_guided_semantic_matching_deepseek.json"

MODEL_NAMES = [
    "Zero-shot",
    "Vanilla SFT",
    "TaxSFT"
]

METRICS = [
    "diagnostic_consistency",
    "evidence_consistency",
    "defect_analysis_match",
    "repair_logic_match",
    "scaffolding_match"
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
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


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
    print("Reference-Guided Semantic Matching Summary")
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
                scores_by_model[model_name][metric].append(scores[key])

    summary = {}

    print("\n" + "=" * 80)
    print("Average RGSM Scores by Model")
    print("=" * 80)

    for model in MODEL_NAMES:
        print(f"\n{model}")

        summary[model] = {}

        metric_means = []
        all_scores = []

        for metric in METRICS:
            values = scores_by_model[model][metric]
            m = mean(values)
            c = ci95(values)

            metric_means.append(m)
            all_scores.extend(values)

            summary[model][metric] = {
                "mean": m,
                "ci95": c,
                "n": len(values)
            }

            print(f"{metric}: {m:.3f} ± {c:.3f} (95% CI)")

        overall = mean(metric_means)
        overall_ci = ci95(all_scores)

        summary[model]["overall"] = {
            "mean": overall,
            "ci95": overall_ci
        }

        print(f"overall: {overall:.3f} ± {overall_ci:.3f} (approx. 95% CI)")

    print("\n" + "=" * 80)
    print("Best Reference Match Count")
    print("=" * 80)

    total = len(data)

    for model in MODEL_NAMES + ["tie"]:
        count = best_counter[model]
        ratio = count / total if total else 0.0
        print(f"{model}: {count}/{total} = {ratio:.2%}")

    summary["_best_reference_match_count"] = dict(best_counter)
    summary["_valid_samples"] = total

    output_file = JUDGE_FILE.replace(".json", "_summary.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f"Summary JSON saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()