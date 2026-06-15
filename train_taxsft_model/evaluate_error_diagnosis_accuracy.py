#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from collections import Counter, defaultdict


# =====================================================
# 1. Configuration
# =====================================================

PREDICTION_FILES = {
    "TaxSFT": "./result_inference/predictions_BioCRDIA-7B-TaxSFT.json",
    "TaxSFT-NeutralInstr": "../abolation_model/result_inference/predictions_BioCRDIA-7B-TaxSFT-NeutralInstr_TaxTest.json",
    # 如果你已经用同一个 tax_test.json 重新推理了其他模型，就打开下面两行
    "VanillaSFT": "../abolation_model/result_inference/predictions_BioCRDIA-7B-VanillaSFT-TaxData_TaxTest.json",
    "Zero-shot": "./result_inference/predictions_Qwen2.5-7B-Zeroshot_TaxTest.json",
    "GPT-4o Zero-shot": "./result_inference/predictions_GPT-4o_Zeroshot_TaxTest.json",
}

VALID_ERROR_TYPES = [
    "NW_BOUNDARY_INITIALIZATION_ERROR",
    "NW_GAP_PENALTY_SIGN_ERROR",
    "NW_MISMATCH_SCORING_ERROR",

    "PWM_LOG_ZERO_NO_PSEUDOCOUNT",
    "PWM_PSEUDOCOUNT_DENOMINATOR_ERROR",
    "PWM_BACKGROUND_PROBABILITY_ERROR",

    "TRANSLATION_STOP_CODON_HANDLING_ERROR",
    "TRANSLATION_READING_FRAME_ERROR",
    "MUTATION_EFFECT_MISCLASSIFICATION",
]


# =====================================================
# 2. Error Type Extraction
# =====================================================

def extract_error_type(prediction):
    """
    从模型输出中抽取 Error Type。
    优先匹配：
    Error Type: XXX

    如果没有标准格式，则尝试在全文中搜索合法标签。
    """

    if prediction is None:
        return "MISSING"

    text = prediction.strip()

    # 1. 标准格式抽取
    patterns = [
        r"Error\s*Type\s*[:：]\s*([A-Z0-9_]+)",
        r"\*\*Error\s*Type:\*\*\s*([A-Z0-9_]+)",
        r"error_type\s*[:：]\s*([A-Z0-9_]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            candidate = match.group(1).strip().upper()

            if candidate in VALID_ERROR_TYPES:
                return candidate

    # 2. 兜底：全文搜索合法标签
    for label in VALID_ERROR_TYPES:
        if label in text:
            return label

    return "MISSING"


def get_prediction_text(item):
    if "prediction" in item:
        return item["prediction"]
    if "model_prediction" in item:
        return item["model_prediction"]
    return ""


# =====================================================
# 3. Evaluation
# =====================================================

def evaluate_one_file(model_name, file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = 0
    correct = 0
    missing = 0

    per_type_total = Counter()
    per_type_correct = Counter()

    confusion = defaultdict(Counter)

    error_cases = []

    for item in data:
        gold = item["error_type"]
        prediction_text = get_prediction_text(item)

        pred = extract_error_type(prediction_text)

        total += 1
        per_type_total[gold] += 1
        confusion[gold][pred] += 1

        if pred == "MISSING":
            missing += 1

        if pred == gold:
            correct += 1
            per_type_correct[gold] += 1
        else:
            error_cases.append({
                "id": item.get("id", ""),
                "gold": gold,
                "pred": pred,
                "prediction_preview": prediction_text[:500]
            })

    accuracy = correct / total if total > 0 else 0

    print("\n" + "=" * 80)
    print(f"Model: {model_name}")
    print("=" * 80)
    print(f"File: {file_path}")
    print(f"Total:   {total}")
    print(f"Correct: {correct}")
    print(f"Missing label: {missing}")
    print(f"Error Diagnosis Accuracy: {accuracy:.4f} = {accuracy:.2%}")

    print("\nPer-error-type Accuracy")
    print("-" * 80)

    for error_type in VALID_ERROR_TYPES:
        t = per_type_total[error_type]
        c = per_type_correct[error_type]

        if t == 0:
            continue

        acc = c / t
        print(f"{error_type}: {c}/{t} = {acc:.2%}")

    # 保存错误样本，方便人工查看
    error_file = f"./result_inference/error_diagnosis_wrong_{model_name}.json"

    with open(error_file, "w", encoding="utf-8") as f:
        json.dump(error_cases, f, ensure_ascii=False, indent=2)

    print(f"\nWrong cases saved to: {error_file}")

    return {
        "model_name": model_name,
        "total": total,
        "correct": correct,
        "missing": missing,
        "accuracy": accuracy,
        "per_type_total": dict(per_type_total),
        "per_type_correct": dict(per_type_correct)
    }


def main():
    summary = []

    for model_name, file_path in PREDICTION_FILES.items():
        result = evaluate_one_file(model_name, file_path)
        summary.append(result)

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    for item in summary:
        print(
            f"{item['model_name']}: "
            f"{item['correct']}/{item['total']} = {item['accuracy']:.2%}, "
            f"missing={item['missing']}"
        )


if __name__ == "__main__":
    main()