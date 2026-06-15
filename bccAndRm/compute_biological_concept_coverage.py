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

MODEL_FILES = {
    "Zero-shot": "../train_taxsft_model/result_inference/predictions_Qwen2.5-7B-Zeroshot_TaxTest.json",
    "Vanilla SFT": "../abolation_model/result_inference/predictions_BioCRDIA-7B-VanillaSFT-TaxData_TaxTest.json",
    "TaxSFT": "../train_taxsft_model/result_inference/predictions_BioCRDIA-7B-TaxSFT.json"
}


OUTPUT_DIR = "./result_data"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "biological_concept_coverage_summary.json"
)


# =====================================================
# 2. Concept Sets
# =====================================================
# 每个 error type 定义若干 concept groups。
# 每个 group 中只要命中一个关键词/正则，就算该 concept group covered。
# 单条样本 BCC = covered groups / total groups。

CONCEPT_GROUPS = {
    "NW_BOUNDARY_INITIALIZATION_ERROR": [
        [
            r"boundary", r"initiali[sz]e", r"initiali[sz]ation",
            r"first row", r"first column"
        ],
        [
            r"cumulative gap", r"gap penalt", r"i\s*\*\s*gap",
            r"j\s*\*\s*gap"
        ],
        [
            r"global alignment", r"Needleman", r"full[- ]length",
            r"leading gap", r"trailing gap"
        ],
        [
            r"zero initiali[sz]ation", r"free gap", r"local alignment",
            r"semi[- ]global"
        ]
    ],

    "NW_GAP_PENALTY_SIGN_ERROR": [
        [
            r"gap penalt", r"gap cost", r"gap score"
        ],
        [
            r"wrong sign", r"sign error", r"positive", r"negative",
            r"reward", r"penal"
        ],
        [
            r"excessive gap", r"many gaps", r"insert gaps",
            r"artificially high", r"inflated score"
        ],
        [
            r"Needleman", r"global alignment", r"dynamic programming"
        ]
    ],

    "NW_MISMATCH_SCORING_ERROR": [
        [
            r"mismatch", r"substitution score", r"scoring parameter"
        ],
        [
            r"-5", r"penalty", r"zero", r"0", r"incorrect score"
        ],
        [
            r"diagonal", r"recurrence", r"dynamic programming",
            r"matrix"
        ],
        [
            r"wrong path", r"traceback", r"inflated score",
            r"prefer mismatches"
        ]
    ],

    "PWM_LOG_ZERO_NO_PSEUDOCOUNT": [
        [
            r"zero count", r"zero probability", r"probability of zero",
            r"0/12", r"unseen nucleotide"
        ],
        [
            r"log2\(0\)", r"log_2\(0\)", r"logarithm of zero",
            r"math domain error", r"undefined"
        ],
        [
            r"pseudocount", r"Laplace", r"smoothing", r"add[- ]one"
        ],
        [
            r"PWM", r"position weight matrix", r"log[- ]odds"
        ]
    ],

    "PWM_PSEUDOCOUNT_DENOMINATOR_ERROR": [
        [
            r"pseudocount", r"Laplace", r"smoothing"
        ],
        [
            r"denominator", r"N\s*\+\s*4", r"total\s*\+\s*4",
            r"N\+4", r"16"
        ],
        [
            r"probabilities sum", r"sum to 1", r"normalisation",
            r"normalization", r"valid probability distribution"
        ],
        [
            r"numerator", r"count\s*\+\s*1", r"pseudocount mass"
        ]
    ],

    "PWM_BACKGROUND_PROBABILITY_ERROR": [
        [
            r"background", r"background probability", r"0\.25",
            r"uniform background"
        ],
        [
            r"log[- ]odds", r"log2\(.*\/.*0\.25", r"log_2\(.*\/.*0\.25",
            r"probability.*background"
        ],
        [
            r"log2\(p\)", r"log_2\(p\)", r"log probability",
            r"omitted.*background", r"forgot.*background"
        ],
        [
            r"enrichment", r"depletion", r"relative to chance",
            r"null model"
        ]
    ],

    "TRANSLATION_STOP_CODON_HANDLING_ERROR": [
        [
            r"stop codon", r"TAA", r"TAG", r"TGA"
        ],
        [
            r"terminate", r"termination", r"stop translation",
            r"first in[- ]frame stop"
        ],
        [
            r"premature stop", r"truncated", r"shorter protein"
        ],
        [
            r"nonsense", r"premature termination"
        ]
    ],

    "TRANSLATION_READING_FRAME_ERROR": [
        [
            r"reading frame", r"frame", r"codon frame"
        ],
        [
            r"point substitution", r"single[- ]nucleotide substitution",
            r"substitution does not", r"does not shift"
        ],
        [
            r"codon boundar", r"triplet", r"group.*three",
            r"start from the first nucleotide"
        ],
        [
            r"frameshift", r"artificial frameshift", r"wrong offset"
        ]
    ],

    "MUTATION_EFFECT_MISCLASSIFICATION": [
        [
            r"mutation effect", r"classification", r"classify"
        ],
        [
            r"silent", r"missense", r"nonsense"
        ],
        [
            r"protein[- ]level", r"amino acid sequence",
            r"translated protein", r"protein consequence"
        ],
        [
            r"same amino acid", r"different amino acid",
            r"stop codon", r"truncated"
        ]
    ]
}


ERROR_DOMAIN = {
    "NW_BOUNDARY_INITIALIZATION_ERROR": "Alignment",
    "NW_GAP_PENALTY_SIGN_ERROR": "Alignment",
    "NW_MISMATCH_SCORING_ERROR": "Alignment",

    "PWM_LOG_ZERO_NO_PSEUDOCOUNT": "PWM",
    "PWM_PSEUDOCOUNT_DENOMINATOR_ERROR": "PWM",
    "PWM_BACKGROUND_PROBABILITY_ERROR": "PWM",

    "TRANSLATION_STOP_CODON_HANDLING_ERROR": "Mutation",
    "TRANSLATION_READING_FRAME_ERROR": "Mutation",
    "MUTATION_EFFECT_MISCLASSIFICATION": "Mutation"
}


# =====================================================
# 3. Utility
# =====================================================

def get_prediction(item):
    if "prediction" in item:
        return item["prediction"]
    if "model_prediction" in item:
        return item["model_prediction"]
    return ""


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


def match_group(text, patterns):
    text = text.lower()
    for pattern in patterns:
        if re.search(pattern.lower(), text, flags=re.IGNORECASE):
            return True
    return False


def compute_bcc_for_item(prediction, error_type):
    groups = CONCEPT_GROUPS.get(error_type, [])

    if not groups:
        return 0.0, 0, 0

    covered = 0

    for group in groups:
        if match_group(prediction, group):
            covered += 1

    score = covered / len(groups)

    return score, covered, len(groups)


# =====================================================
# 4. Main
# =====================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary = {}

    print("=" * 80)
    print("Biological Concept Coverage")
    print("=" * 80)

    for model_name, path in MODEL_FILES.items():
        print(f"\nModel: {model_name}")
        print(f"File: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        overall_scores = []
        domain_scores = defaultdict(list)
        error_type_scores = defaultdict(list)

        detailed = []

        for item in data:
            error_type = item["error_type"]
            domain = ERROR_DOMAIN.get(error_type, item.get("domain", "Unknown"))
            prediction = get_prediction(item)

            score, covered, total = compute_bcc_for_item(
                prediction=prediction,
                error_type=error_type
            )

            overall_scores.append(score)
            domain_scores[domain].append(score)
            error_type_scores[error_type].append(score)

            detailed.append({
                "id": item["id"],
                "model_name": model_name,
                "error_type": error_type,
                "domain": domain,
                "coverage_score": score,
                "covered_concepts": covered,
                "total_concepts": total
            })

        model_summary = {
            "overall": {
                "mean": mean(overall_scores),
                "ci95": ci95(overall_scores),
                "n": len(overall_scores)
            },
            "by_domain": {},
            "by_error_type": {}
        }

        print(f"Overall BCC: {mean(overall_scores) * 100:.2f}% ± {ci95(overall_scores) * 100:.2f}%")

        print("By domain:")
        for domain, values in domain_scores.items():
            model_summary["by_domain"][domain] = {
                "mean": mean(values),
                "ci95": ci95(values),
                "n": len(values)
            }
            print(f"  {domain}: {mean(values) * 100:.2f}% ± {ci95(values) * 100:.2f}%")

        print("By error type:")
        for error_type, values in error_type_scores.items():
            model_summary["by_error_type"][error_type] = {
                "mean": mean(values),
                "ci95": ci95(values),
                "n": len(values)
            }
            print(f"  {error_type}: {mean(values) * 100:.2f}%")

        summary[model_name] = model_summary

        detailed_file = os.path.join(
            OUTPUT_DIR,
            f"biological_concept_coverage_details_{model_name.replace(' ', '_')}.json"
        )

        with open(detailed_file, "w", encoding="utf-8") as f:
            json.dump(detailed, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f"Summary saved to: {OUTPUT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()