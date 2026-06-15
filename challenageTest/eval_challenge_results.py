#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import math
from collections import defaultdict


# =====================================================
# 1. Configuration
# =====================================================

# 改成你截图里的结果目录
RESULT_DIR = "/root/autodl-tmp/APBC_vgpu/challenageTest/result_data/challenge_infer_apbc_hard"

OUTPUT_DIR = os.path.join(RESULT_DIR, "eval_outputs")

PREDICTION_FILES = {
    "Qwen2.5-7B Zero-shot": "predictions_Qwen2.5-7B-ZeroShot_challenge.json",
    "Vanilla SFT": "predictions_BioCRDIA-7B-VanillaSFT-TaxData_challenge.json",
    "TaxSFT-NeutralInstr": "predictions_BioCRDIA-7B-TaxSFT-NeutralInstr_challenge.json",
    "TaxSFT": "predictions_BioCRDIA-7B-TaxSFT_challenge.json",
}


ERROR_TYPES = [
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


SHORT_NAMES = {
    "NW_BOUNDARY_INITIALIZATION_ERROR": "E1",
    "NW_GAP_PENALTY_SIGN_ERROR": "E2",
    "NW_MISMATCH_SCORING_ERROR": "E3",
    "PWM_LOG_ZERO_NO_PSEUDOCOUNT": "M1",
    "PWM_PSEUDOCOUNT_DENOMINATOR_ERROR": "M2",
    "PWM_BACKGROUND_PROBABILITY_ERROR": "M3",
    "TRANSLATION_STOP_CODON_HANDLING_ERROR": "T1",
    "TRANSLATION_READING_FRAME_ERROR": "T2",
    "MUTATION_EFFECT_MISCLASSIFICATION": "T3",
}


# 如果你要在表里显示 standard test 的原始结果，可以填这里。
# 没有就保持 None。
STANDARD_EDA = {
    "Qwen2.5-7B Zero-shot": 49.33,
    "Vanilla SFT": 49.11,
    "TaxSFT-NeutralInstr": 59.33,
    "TaxSFT": 97.11,
}


# =====================================================
# 2. Label extraction
# =====================================================

def extract_error_type(prediction: str):
    """
    从模型输出中抽取 Error Type。
    优先抽取 'Error Type: XXX'，失败后全文搜索合法标签。
    """
    if prediction is None:
        return None

    text = prediction.strip()

    # 1. 优先匹配 Error Type 字段
    patterns = [
        r"Error\s*Type\s*:\s*([A-Z0-9_]+)",
        r"Error\s*type\s*:\s*([A-Z0-9_]+)",
        r"Diagnostic\s*Label\s*:\s*([A-Z0-9_]+)",
        r"Diagnostic\s*label\s*:\s*([A-Z0-9_]+)",
    ]

    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            label = m.group(1).strip()
            if label in ERROR_TYPES:
                return label

    # 2. 兜底：全文搜索合法 taxonomy label
    for label in ERROR_TYPES:
        if label in text:
            return label

    return None


# =====================================================
# 3. Evaluation
# =====================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict) and "data" in data:
        return data["data"]

    raise ValueError(f"Unsupported JSON format: {path}")


def evaluate_model(model_name, file_path):
    data = load_json(file_path)

    total = len(data)
    correct = 0
    missing = 0

    per_type = defaultdict(lambda: {"total": 0, "correct": 0, "missing": 0})
    detailed = []
    wrong_cases = []

    for item in data:
        gold = item.get("error_type", "")
        prediction = item.get("prediction", "")
        pred_label = extract_error_type(prediction)

        is_missing = pred_label is None
        is_correct = pred_label == gold

        correct += int(is_correct)
        missing += int(is_missing)

        per_type[gold]["total"] += 1
        per_type[gold]["correct"] += int(is_correct)
        per_type[gold]["missing"] += int(is_missing)

        row = {
            "id": item.get("id", ""),
            "model_name": model_name,
            "domain": item.get("domain", ""),
            "gold_error_type": gold,
            "pred_error_type": pred_label,
            "correct": is_correct,
            "missing_label": is_missing,
            "sub_error_type": item.get("sub_error_type", ""),
            "input": item.get("input", ""),
            "reference_solution": item.get("reference_solution", ""),
            "prediction": prediction,
        }

        detailed.append(row)

        if not is_correct:
            wrong_cases.append(row)

    eda = correct / total if total > 0 else 0.0
    mlr = missing / total if total > 0 else 0.0

    return {
        "model_name": model_name,
        "total": total,
        "correct": correct,
        "eda": eda,
        "missing": missing,
        "mlr": mlr,
        "per_type": per_type,
        "detailed": detailed,
        "wrong_cases": wrong_cases,
    }


# =====================================================
# 4. McNemar test
# =====================================================

def exact_mcnemar_p_value(b01, b10):
    """
    Exact binomial McNemar test.
    b01: model A correct, model B wrong
    b10: model A wrong, model B correct
    """
    n = b01 + b10
    if n == 0:
        return 1.0

    k = min(b01, b10)
    prob = 0.0
    for i in range(0, k + 1):
        prob += math.comb(n, i) * (0.5 ** n)

    return min(1.0, 2 * prob)


def mcnemar_against_taxsft(all_results):
    tax_detail = all_results["TaxSFT"]["detailed"]
    tax_map = {x["id"]: x for x in tax_detail}

    rows = []

    for model_name, result in all_results.items():
        if model_name == "TaxSFT":
            continue

        other_map = {x["id"]: x for x in result["detailed"]}

        tax_correct_other_wrong = 0
        tax_wrong_other_correct = 0

        for sid, tax_row in tax_map.items():
            if sid not in other_map:
                continue

            tax_correct = tax_row["correct"]
            other_correct = other_map[sid]["correct"]

            if tax_correct and not other_correct:
                tax_correct_other_wrong += 1
            elif (not tax_correct) and other_correct:
                tax_wrong_other_correct += 1

        p_value = exact_mcnemar_p_value(
            tax_correct_other_wrong,
            tax_wrong_other_correct
        )

        rows.append({
            "comparison": f"TaxSFT vs {model_name}",
            "taxsft_correct_other_wrong": tax_correct_other_wrong,
            "taxsft_wrong_other_correct": tax_wrong_other_correct,
            "p_value": p_value,
        })

    return rows


# =====================================================
# 5. Output utilities
# =====================================================

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def make_summary_table(all_results):
    rows = []

    for model_name, result in all_results.items():
        standard = STANDARD_EDA.get(model_name, None)
        challenge = result["eda"] * 100
        mlr = result["mlr"] * 100

        if standard is None:
            drop = None
        else:
            drop = standard - challenge

        rows.append({
            "Model": model_name,
            "Standard_EDA": standard,
            "Challenge_EDA": challenge,
            "Drop": drop,
            "Missing": result["missing"],
            "MLR": mlr,
            "Correct": result["correct"],
            "Total": result["total"],
        })

    return rows


def print_summary(summary_rows):
    print("=" * 100)
    print("Challenge Set Overall Results")
    print("=" * 100)
    print(f"{'Model':30s} {'Std EDA':>10s} {'Chal EDA':>10s} {'Drop':>10s} {'Missing':>8s} {'MLR':>8s}")

    for r in summary_rows:
        std = "--" if r["Standard_EDA"] is None else f"{r['Standard_EDA']:.2f}"
        drop = "--" if r["Drop"] is None else f"{r['Drop']:.2f}"
        print(
            f"{r['Model']:30s} "
            f"{std:>10s} "
            f"{r['Challenge_EDA']:10.2f} "
            f"{drop:>10s} "
            f"{r['Missing']:8d} "
            f"{r['MLR']:7.2f}%"
        )

    print("=" * 100)


def make_latex_table(summary_rows):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Robustness on the template-disjoint challenge set.}")
    lines.append(r"\label{tab:challenge_test}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\hline")
    lines.append(r"Model & Standard EDA & Challenge EDA & Drop & MLR \\")
    lines.append(r"\hline")

    for r in summary_rows:
        std = "--" if r["Standard_EDA"] is None else f"{r['Standard_EDA']:.2f}"
        chal = f"{r['Challenge_EDA']:.2f}"
        drop = "--" if r["Drop"] is None else f"{r['Drop']:.2f}"
        mlr = f"{r['MLR']:.2f}"

        lines.append(
            f"{r['Model']} & {std} & {chal} & {drop} & {mlr} \\\\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def make_per_type_latex(all_results):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-error-type EDA on the template-disjoint challenge set.}")
    lines.append(r"\label{tab:challenge_per_type}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\hline")
    lines.append(r"Error & Zero-shot & Vanilla SFT & NeutralInstr & TaxSFT \\")
    lines.append(r"\hline")

    name_map = {
        "Qwen2.5-7B Zero-shot": "Zero-shot",
        "Vanilla SFT": "Vanilla SFT",
        "TaxSFT-NeutralInstr": "NeutralInstr",
        "TaxSFT": "TaxSFT",
    }

    model_order = [
        "Qwen2.5-7B Zero-shot",
        "Vanilla SFT",
        "TaxSFT-NeutralInstr",
        "TaxSFT",
    ]

    for error_type in ERROR_TYPES:
        vals = []

        for model_name in model_order:
            result = all_results[model_name]
            item = result["per_type"][error_type]
            total = item["total"]
            correct = item["correct"]
            acc = 100 * correct / total if total > 0 else 0.0
            vals.append(f"{acc:.1f}")

        lines.append(
            f"{SHORT_NAMES[error_type]} & {vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} \\\\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# =====================================================
# 6. Main
# =====================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = {}

    for model_name, filename in PREDICTION_FILES.items():
        file_path = os.path.join(RESULT_DIR, filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Prediction file not found: {file_path}")

        result = evaluate_model(model_name, file_path)
        all_results[model_name] = result

        save_json(
            os.path.join(OUTPUT_DIR, f"wrong_cases_{model_name.replace(' ', '_')}.json"),
            result["wrong_cases"]
        )

        save_json(
            os.path.join(OUTPUT_DIR, f"detailed_{model_name.replace(' ', '_')}.json"),
            result["detailed"]
        )

    summary_rows = make_summary_table(all_results)
    print_summary(summary_rows)

    save_json(os.path.join(OUTPUT_DIR, "challenge_summary.json"), summary_rows)

    mcnemar_rows = mcnemar_against_taxsft(all_results)
    save_json(os.path.join(OUTPUT_DIR, "mcnemar_against_taxsft.json"), mcnemar_rows)

    print("\nMcNemar exact test against TaxSFT:")
    for r in mcnemar_rows:
        print(
            f"{r['comparison']}: "
            f"TaxSFT correct / other wrong = {r['taxsft_correct_other_wrong']}, "
            f"TaxSFT wrong / other correct = {r['taxsft_wrong_other_correct']}, "
            f"p = {r['p_value']:.6g}"
        )

    latex_table = make_latex_table(summary_rows)
    with open(os.path.join(OUTPUT_DIR, "challenge_table.tex"), "w", encoding="utf-8") as f:
        f.write(latex_table)

    per_type_latex = make_per_type_latex(all_results)
    with open(os.path.join(OUTPUT_DIR, "challenge_per_type_table.tex"), "w", encoding="utf-8") as f:
        f.write(per_type_latex)

    print("\nLaTeX table:")
    print("-" * 100)
    print(latex_table)
    print("-" * 100)

    print(f"\nSaved evaluation outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()