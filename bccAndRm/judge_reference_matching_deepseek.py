#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import random
import hashlib
import re
from tqdm import tqdm
from openai import OpenAI


# =====================================================
# 1. Configuration
# =====================================================

DEEPSEEK_API_KEY = ""

BASE_URL = "https://api.deepseek.com"
JUDGE_MODEL = "deepseek-chat"

ZERO_SHOT_FILE = "../train_taxsft_model/result_inference/predictions_Qwen2.5-7B-Zeroshot_TaxTest.json"

VANILLA_FILE = "../abolation_model/result_inference/predictions_BioCRDIA-7B-VanillaSFT-TaxData_TaxTest.json"

TAXSFT_FILE = "../train_taxsft_model/result_inference/predictions_BioCRDIA-7B-TaxSFT.json"

MODEL_FILES = {
    "Zero-shot": ZERO_SHOT_FILE,
    "Vanilla SFT": VANILLA_FILE,
    "TaxSFT": TAXSFT_FILE
}

OUTPUT_DIR = "./result_data"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "judge_reference_guided_semantic_matching_deepseek.json"
)

SEED = 42
MAX_RETRY = 3
SLEEP_TIME = 1.0

SAMPLE_LIMIT = None

MAX_INPUT_CHARS = 6000
MAX_REFERENCE_CHARS = 6000
MAX_RESPONSE_CHARS = 6000


client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=BASE_URL
)


# =====================================================
# 2. Utility
# =====================================================

def stable_int_hash(value):
    value = str(value).encode("utf-8")
    return int(hashlib.md5(value).hexdigest(), 16)


def truncate_text(text, max_chars):
    if text is None:
        return ""
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[TRUNCATED]"


def safe_json_loads(text):
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"^```", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

    return json.loads(text)


def get_prediction(item):
    if "prediction" in item:
        return item["prediction"]
    if "model_prediction" in item:
        return item["model_prediction"]
    return ""


def load_prediction_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {str(item["id"]): item for item in data}


def assign_blind_responses(sample_id, model_to_prediction):
    rng = random.Random(SEED + stable_int_hash(sample_id))

    model_names = list(model_to_prediction.keys())
    rng.shuffle(model_names)

    sides = ["A", "B", "C"]

    side_to_response = {}
    side_to_model = {}

    for side, model_name in zip(sides, model_names):
        side_to_model[side] = model_name
        side_to_response[side] = model_to_prediction[model_name]

    return side_to_response, side_to_model


def validate_result(result):
    metrics = [
        "diagnostic_consistency",
        "evidence_consistency",
        "defect_analysis_match",
        "repair_logic_match",
        "scaffolding_match"
    ]

    for side in ["A", "B", "C"]:
        for metric in metrics:
            key = f"{metric}_{side}"
            if key not in result:
                return False, f"Missing key: {key}"

            value = result[key]
            if not isinstance(value, int):
                return False, f"Non-integer score: {key}={value}"

            if value < 1 or value > 5:
                return False, f"Score out of range: {key}={value}"

    if "best_reference_match" not in result:
        return False, "Missing key: best_reference_match"

    if result["best_reference_match"] not in ["A", "B", "C", "tie"]:
        return False, f"Invalid best_reference_match: {result['best_reference_match']}"

    if "brief_reason" not in result:
        return False, "Missing key: brief_reason"

    return True, "OK"


# =====================================================
# 3. Prompt
# =====================================================

def build_prompt(sample, side_to_response):
    sample_input = truncate_text(sample["input"], MAX_INPUT_CHARS)
    reference = truncate_text(sample["ground_truth"], MAX_REFERENCE_CHARS)
    gold_error_type = sample["error_type"]

    response_a = truncate_text(side_to_response["A"], MAX_RESPONSE_CHARS)
    response_b = truncate_text(side_to_response["B"], MAX_RESPONSE_CHARS)
    response_c = truncate_text(side_to_response["C"], MAX_RESPONSE_CHARS)

    return f"""
You are an expert evaluator for bioinformatics intelligent tutoring systems.

Your task is to evaluate how semantically well each model response matches the reference tutoring feedback.

The reference feedback is teacher-distilled and should be treated as factual guidance.
Do NOT evaluate by lexical overlap. Different wording is acceptable if the diagnostic meaning and pedagogical content are preserved.

Evaluate each response independently on the following five dimensions:

1. Diagnostic Consistency:
Whether the response identifies the same underlying error type or misconception as the reference.

2. Evidence Consistency:
Whether the response cites evidence from the student's execution state that supports the same diagnosis as the reference.

3. Defect Analysis Match:
Whether the response explains the same root cause as the reference.

4. Repair Logic Match:
Whether the response gives a repair strategy consistent with the reference.

5. Scaffolding Match:
Whether the response provides a learning-oriented strategy consistent with the reference.

Scoring rules:
- Use a 1 to 5 integer score for each dimension.
- 5 = semantically equivalent to the reference and complete.
- 4 = mostly aligned with minor omissions.
- 3 = partially aligned but incomplete or generic.
- 2 = weakly aligned or contains important inconsistencies.
- 1 = inconsistent with the reference or misleading.
- Do not prefer a response just because it is longer.
- Do not prefer a response just because it sounds more fluent.
- Penalize hallucinated biological, computational, or numerical claims.
- Use the gold error type only as factual guidance.

Return your evaluation strictly in JSON format only.

The JSON schema must be:
{{
  "diagnostic_consistency_A": 1-5,
  "evidence_consistency_A": 1-5,
  "defect_analysis_match_A": 1-5,
  "repair_logic_match_A": 1-5,
  "scaffolding_match_A": 1-5,

  "diagnostic_consistency_B": 1-5,
  "evidence_consistency_B": 1-5,
  "defect_analysis_match_B": 1-5,
  "repair_logic_match_B": 1-5,
  "scaffolding_match_B": 1-5,

  "diagnostic_consistency_C": 1-5,
  "evidence_consistency_C": 1-5,
  "defect_analysis_match_C": 1-5,
  "repair_logic_match_C": 1-5,
  "scaffolding_match_C": 1-5,

  "best_reference_match": "A" or "B" or "C" or "tie",
  "brief_reason": "one or two sentences explaining the decision"
}}

[Problem and Student Execution State]
{sample_input}

[Gold Error Type]
{gold_error_type}

[Reference Feedback]
{reference}

[Response A]
{response_a}

[Response B]
{response_b}

[Response C]
{response_c}
""".strip()


# =====================================================
# 4. DeepSeek API Call
# =====================================================

def call_deepseek(prompt):
    for attempt in range(MAX_RETRY):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict academic evaluator. Output valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            result = safe_json_loads(content)

            valid, msg = validate_result(result)
            if not valid:
                raise ValueError(f"Invalid result: {msg}. Raw: {content}")

            return result

        except Exception as e:
            print(f"[Retry {attempt + 1}/{MAX_RETRY}] Error: {e}")
            time.sleep(SLEEP_TIME * (attempt + 1))

    return None


# =====================================================
# 5. Main
# =====================================================

def main():
    if DEEPSEEK_API_KEY.startswith("sk-你的") or not DEEPSEEK_API_KEY.strip():
        raise ValueError("Please set your valid DeepSeek API key in DEEPSEEK_API_KEY.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("Reference-Guided Semantic Matching with DeepSeek")
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 80)

    model_data = {}

    for model_name, path in MODEL_FILES.items():
        print(f"Loading {model_name}: {path}")
        model_data[model_name] = load_prediction_file(path)

    common_ids = None
    for id_to_item in model_data.values():
        ids = set(id_to_item.keys())
        common_ids = ids if common_ids is None else common_ids & ids

    common_ids = sorted(
        list(common_ids),
        key=lambda x: stable_int_hash(x)
    )

    if SAMPLE_LIMIT is not None:
        common_ids = common_ids[:SAMPLE_LIMIT]

    print(f"Common samples to evaluate: {len(common_ids)}")

    results = []

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"Loaded existing results: {len(results)}")

    finished_ids = {
        str(item["id"])
        for item in results
        if item.get("status") == "success"
    }

    for sample_id in tqdm(common_ids):
        if sample_id in finished_ids:
            continue

        source_item = model_data["TaxSFT"][sample_id]

        sample = {
            "id": source_item["id"],
            "input": source_item["input"],
            "ground_truth": source_item.get("ground_truth", source_item.get("output", "")),
            "error_type": source_item["error_type"]
        }

        model_to_prediction = {
            model_name: get_prediction(model_data[model_name][sample_id])
            for model_name in MODEL_FILES.keys()
        }

        side_to_response, side_to_model = assign_blind_responses(
            sample_id=sample_id,
            model_to_prediction=model_to_prediction
        )

        prompt = build_prompt(
            sample=sample,
            side_to_response=side_to_response
        )

        judge_result = call_deepseek(prompt)

        if judge_result is None:
            results.append({
                "id": sample["id"],
                "status": "failed",
                "mapping": side_to_model
            })
        else:
            best_raw = judge_result.get("best_reference_match", "tie")

            if best_raw in ["A", "B", "C"]:
                best_model = side_to_model[best_raw]
            else:
                best_model = "tie"

            results.append({
                "id": sample["id"],
                "status": "success",
                "judge_type": "reference_guided_semantic_matching",
                "mapping": side_to_model,
                "best_reference_match_raw": best_raw,
                "best_model": best_model,
                "scores": judge_result
            })

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        time.sleep(SLEEP_TIME)

    print("=" * 80)
    print(f"Finished. Results saved to: {OUTPUT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()