import os
import json
import time
import random
import re
from tqdm import tqdm
from openai import OpenAI


# =====================================================
# 1. Configuration
# =====================================================

DEEPSEEK_API_KEY = ""
BASE_URL = "https://api.deepseek.com"
JUDGE_MODEL = "deepseek-chat"

ZERO_SHOT_FILE = "./result_inference/predictions_Qwen2.5-7B-Zeroshot.json"
RPD_FILE = "./result_inference/predictions_BioCRDIA-7B-RPD-0.3.json"

OUTPUT_FILE = "./result_data/judge_no_reference_rpd03_vs_zeroshot.json"

TARGET_MODEL = "BioCRDIA-7B-RPD-0.3"
BASELINE_MODEL = "zero-shot"

SEED = 42
MAX_RETRY = 3
SLEEP_TIME = 1.0


client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=BASE_URL
)


# =====================================================
# 2. Utility
# =====================================================

def safe_json_loads(text):
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```json", "", text)
        text = re.sub(r"^```", "", text)
        text = re.sub(r"```$", "", text)
        text = text.strip()

    return json.loads(text)


def get_prediction(item):
    if "prediction" in item:
        return item["prediction"]
    if "model_prediction" in item:
        return item["model_prediction"]
    raise KeyError("No prediction field found.")


def assign_blind_pair(zero_pred, rpd_pred, sample_id):
    rng = random.Random(SEED + int(sample_id))

    if rng.random() < 0.5:
        response_a = zero_pred
        response_b = rpd_pred
        mapping = {
            "A": BASELINE_MODEL,
            "B": TARGET_MODEL
        }
    else:
        response_a = rpd_pred
        response_b = zero_pred
        mapping = {
            "A": TARGET_MODEL,
            "B": BASELINE_MODEL
        }

    return response_a, response_b, mapping


# =====================================================
# 3. No-reference Judge Prompt
# =====================================================

def build_no_reference_prompt(sample, response_a, response_b):
    return f"""
You are an expert evaluator for bioinformatics intelligent tutoring systems.

Your task is to compare two model responses to the same student problem.
You are NOT given a reference answer.
You must evaluate the responses based only on the problem, the student's execution state, and the factual correctness of each response.

Evaluate which response provides better process-oriented pedagogical feedback.

You must judge based on the following four dimensions:

1. Diagnosis Accuracy:
Whether the response correctly identifies the likely root cause of the student's error from the execution state.

2. Repair Correctness:
Whether the response provides a correct, actionable, and case-specific repair strategy.

3. Bioinformatics Relevance:
Whether the response correctly uses relevant bioinformatics concepts, algorithms, and biological interpretation.

4. Pedagogical Quality:
Whether the response provides useful learning guidance, conceptual explanation, and scaffolding.

Important rules:
- Do not prefer a response just because it is longer.
- Do not prefer a response just because it uses headings or a cleaner structure.
- Penalize hallucinated biological, computational, or algorithmic claims.
- Penalize responses that ignore the student's execution state.
- Penalize responses that merely solve the original problem without diagnosing the student's error.
- Penalize any incorrect numerical claim, especially incorrect alignment scores, probabilities, protein sequences, matrix values, or final scores.
- Penalize vague advice that is not connected to the observed error.
- If both responses are similarly good, choose "tie".

The winner should be based primarily on factual correctness, execution-state diagnosis, and teaching usefulness.

Return your evaluation strictly in JSON format only.

The JSON schema must be:
{{
  "diagnosis_accuracy_A": 1-5,
  "diagnosis_accuracy_B": 1-5,
  "repair_correctness_A": 1-5,
  "repair_correctness_B": 1-5,
  "bioinformatics_relevance_A": 1-5,
  "bioinformatics_relevance_B": 1-5,
  "pedagogical_quality_A": 1-5,
  "pedagogical_quality_B": 1-5,
  "critical_errors_A": ["list major factual or numerical errors, or empty list"],
  "critical_errors_B": ["list major factual or numerical errors, or empty list"],
  "winner": "A" or "B" or "tie",
  "brief_reason": "one or two sentences explaining the decision"
}}

[Problem and Student Execution State]
{sample["input"]}

[Response A]
{response_a}

[Response B]
{response_b}
""".strip()


# =====================================================
# 4. DeepSeek API Call
# =====================================================

def call_deepseek_judge(prompt):
    for attempt in range(MAX_RETRY):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict and fair academic evaluator. Output valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.0,
                response_format={
                    "type": "json_object"
                }
            )

            content = response.choices[0].message.content
            return safe_json_loads(content)

        except Exception as e:
            print(f"[Retry {attempt + 1}/{MAX_RETRY}] Error: {e}")
            time.sleep(SLEEP_TIME * (attempt + 1))

    return None


# =====================================================
# 5. Main
# =====================================================

def main():

    if DEEPSEEK_API_KEY is None:
        raise ValueError(
            "DEEPSEEK_API_KEY is not set. Run: export DEEPSEEK_API_KEY='your_key'"
        )

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(ZERO_SHOT_FILE, "r", encoding="utf-8") as f:
        zero_data = json.load(f)

    with open(RPD_FILE, "r", encoding="utf-8") as f:
        rpd_data = json.load(f)

    assert len(zero_data) == len(rpd_data), "Two prediction files have different lengths."

    results = []

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            results = json.load(f)

    finished_ids = {
        item["id"] for item in results
        if item.get("status") == "success"
    }

    for i in tqdm(range(len(rpd_data))):

        zero_item = zero_data[i]
        rpd_item = rpd_data[i]

        assert zero_item["id"] == rpd_item["id"], f"ID mismatch at index {i}"

        sample_id = rpd_item["id"]

        if sample_id in finished_ids:
            continue

        sample = {
            "id": sample_id,
            "input": rpd_item["input"]
        }

        zero_pred = get_prediction(zero_item)
        rpd_pred = get_prediction(rpd_item)

        response_a, response_b, mapping = assign_blind_pair(
            zero_pred=zero_pred,
            rpd_pred=rpd_pred,
            sample_id=sample_id
        )

        prompt = build_no_reference_prompt(
            sample=sample,
            response_a=response_a,
            response_b=response_b
        )

        judge_result = call_deepseek_judge(prompt)

        if judge_result is None:
            results.append({
                "id": sample_id,
                "status": "failed",
                "mapping": mapping
            })
        else:
            winner_raw = judge_result.get("winner", "tie")

            if winner_raw in ["A", "B"]:
                winner_model = mapping[winner_raw]
            else:
                winner_model = "tie"

            results.append({
                "id": sample_id,
                "status": "success",
                "judge_type": "no_reference",
                "mapping": mapping,
                "winner_raw": winner_raw,
                "winner_model": winner_model,
                "scores": judge_result
            })

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        time.sleep(SLEEP_TIME)

    print(f"Saved no-reference judge results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()