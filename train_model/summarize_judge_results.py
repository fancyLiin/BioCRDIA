import json
from collections import Counter

# JUDGE_FILE = "./result_data/judge_pairwise_deepseek_rpd03_vs_zeroshot.json"
# JUDGE_FILE = "./result_data/judge_strict_reference_rpd03_vs_zeroshot.json"
JUDGE_FILE = "./result_data/judge_no_reference_rpd03_vs_zeroshot.json"
TARGET_MODEL = "BioCRDIA-7B-RPD-0.3"
BASELINE_MODEL = "zero-shot"


def avg(values):
    return sum(values) / len(values) if values else 0.0


def main():

    with open(JUDGE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = [x for x in data if x.get("status") == "success"]

    win_counter = Counter()

    target_scores = {
        "diagnosis_accuracy": [],
        "repair_correctness": [],
        "bioinformatics_relevance": [],
        "pedagogical_quality": []
    }

    baseline_scores = {
        "diagnosis_accuracy": [],
        "repair_correctness": [],
        "bioinformatics_relevance": [],
        "pedagogical_quality": []
    }

    for item in data:

        win_counter[item["winner_model"]] += 1

        mapping = item["mapping"]
        scores = item["scores"]

        for side in ["A", "B"]:
            model_name = mapping[side]

            if model_name == TARGET_MODEL:
                target_scores["diagnosis_accuracy"].append(scores[f"diagnosis_accuracy_{side}"])
                target_scores["repair_correctness"].append(scores[f"repair_correctness_{side}"])
                target_scores["bioinformatics_relevance"].append(scores[f"bioinformatics_relevance_{side}"])
                target_scores["pedagogical_quality"].append(scores[f"pedagogical_quality_{side}"])

            elif model_name == BASELINE_MODEL:
                baseline_scores["diagnosis_accuracy"].append(scores[f"diagnosis_accuracy_{side}"])
                baseline_scores["repair_correctness"].append(scores[f"repair_correctness_{side}"])
                baseline_scores["bioinformatics_relevance"].append(scores[f"bioinformatics_relevance_{side}"])
                baseline_scores["pedagogical_quality"].append(scores[f"pedagogical_quality_{side}"])

    total = len(data)

    print("=" * 60)
    print("Pairwise Win Rate")
    print("=" * 60)

    for k, v in win_counter.items():
        print(f"{k}: {v} / {total} = {v / total:.2%}")

    print("\n" + "=" * 60)
    print("Average Scores")
    print("=" * 60)

    print(f"\n{TARGET_MODEL}")
    for metric, values in target_scores.items():
        print(f"{metric}: {avg(values):.3f}")

    print(f"\n{BASELINE_MODEL}")
    for metric, values in baseline_scores.items():
        print(f"{metric}: {avg(values):.3f}")

    target_overall = avg([
        avg(v) for v in target_scores.values()
    ])

    baseline_overall = avg([
        avg(v) for v in baseline_scores.values()
    ])

    print("\nOverall")
    print(f"{TARGET_MODEL}: {target_overall:.3f}")
    print(f"{BASELINE_MODEL}: {baseline_overall:.3f}")


if __name__ == "__main__":
    main()