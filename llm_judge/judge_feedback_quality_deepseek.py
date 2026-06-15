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

# 建议直接写在这里，按你的要求不从外部命令传参
DEEPSEEK_API_KEY = ""

BASE_URL = "https://api.deepseek.com"
JUDGE_MODEL = "deepseek-chat"

ZERO_SHOT_FILE = "../train_taxsft_model/result_inference/predictions_Qwen2.5-7B-Zeroshot_TaxTest.json"

VANILLA_FILE = "../abolation_model/result_inference/predictions_BioCRDIA-7B-VanillaSFT-TaxData_TaxTest.json"

TAXSFT_FILE = "../train_taxsft_model/result_inference/predictions_BioCRDIA-7B-TaxSFT.json"

OUTPUT_DIR = "./result_data"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "judge_feedback_quality_deepseek_zeroshot_vanilla_taxsft.json"
)

MODEL_FILES = {
    "Zero-shot": ZERO_SHOT_FILE,
    "Vanilla SFT": VANILLA_FILE,
    "TaxSFT": TAXSFT_FILE
}

SEED = 42
MAX_RETRY = 3
SLEEP_TIME = 1.0

# 如果想先测试前 10 条，改成 10；正式实验改成 None
SAMPLE_LIMIT = None

# 防止单条样本太长导致上下文过长
MAX_RESPONSE_CHARS = 6000
MAX_REFERENCE_CHARS = 6000
MAX_INPUT_CHARS = 6000


client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=BASE_URL
)


# =====================================================
# 2. Utility Functions
# =====================================================

def stable_int_hash(value):
    """
    用稳定 hash 生成整数，保证不同机器/重跑时 A/B/C 映射一致。
    """
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
    """
    兼容 ```json ... ``` 或其他非纯 JSON 包裹。
    """
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```json", "", text.strip(), flags=re.IGNORECASE)
        text = re.sub(r"^```", "", text.strip())
        text = re.sub(r"```$", "", text.strip())
        text = text.strip()

    # 若模型前后输出了多余文字，尝试截取最外层 JSON
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

    id_to_item = {}
    for item in data:
        sample_id = str(item["id"])
        id_to_item[sample_id] = item

    return id_to_item


def assign_blind_responses(sample_id, model_to_prediction):
    """
    对每条样本固定随机打乱三个模型的响应顺序。
    返回:
    - side_to_response: {"A": "...", "B": "...", "C": "..."}
    - side_to_model: {"A": "TaxSFT", ...}
    """
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


def validate_judge_result(result):
    """
    检查 judge 返回是否包含必要字段。
    """
    required_metrics = [
        "diagnosis_accuracy",
        "repair_correctness",
        "bioinformatics_relevance",
        "pedagogical_quality"
    ]

    sides = ["A", "B", "C"]

    for side in sides:
        for metric in required_metrics:
            key = f"{metric}_{side}"
            if key not in result:
                return False, f"Missing key: {key}"

            value = result[key]
            if not isinstance(value, int):
                return False, f"Non-integer score: {key}={value}"

            if value < 1 or value > 5:
                return False, f"Score out of range: {key}={value}"

    if "best_response" not in result:
        return False, "Missing key: best_response"

    if result["best_response"] not in ["A", "B", "C", "tie"]:
        return False, f"Invalid best_response: {result['best_response']}"

    if "brief_reason" not in result:
        return False, "Missing key: brief_reason"

    return True, "OK"


# =====================================================
# 3. Judge Prompt
# =====================================================

def build_judge_prompt(sample, side_to_response):
    sample_input = truncate_text(sample["input"], MAX_INPUT_CHARS)
    reference = truncate_text(sample["ground_truth"], MAX_REFERENCE_CHARS)
    gold_error_type = sample["error_type"]

    response_a = truncate_text(side_to_response["A"], MAX_RESPONSE_CHARS)
    response_b = truncate_text(side_to_response["B"], MAX_RESPONSE_CHARS)
    response_c = truncate_text(side_to_response["C"], MAX_RESPONSE_CHARS)

    return f"""
You are an expert evaluator for bioinformatics intelligent tutoring systems.

Your task is to evaluate three model responses to the same student problem.
The goal is to assess the quality of process-oriented pedagogical feedback.

Evaluate each response independently on the following four dimensions:

1. Diagnosis Accuracy:
Whether the response correctly identifies the root cause of the student's error and the correct error type.

2. Repair Correctness:
Whether the response provides a correct, actionable, and case-specific repair strategy.

3. Bioinformatics Relevance:
Whether the response correctly uses relevant biological and computational concepts.

4. Pedagogical Quality:
Whether the response provides useful learning guidance, conceptual explanation, and scaffolding.

Scoring rules:
- Use a 1 to 5 integer score for each dimension.
- 5 = excellent, fully correct, specific, and educationally useful.
- 4 = mostly correct, with minor omissions.
- 3 = partially correct but incomplete or somewhat generic.
- 2 = mostly incorrect or weakly related.
- 1 = incorrect, misleading, or fails to address the student's error.
- Do not prefer a response merely because it is longer.
- Do not prefer a response merely because it is more fluent or better formatted.
- Penalize hallucinated biological, computational, or numerical claims.
- Penalize responses that miss the actual execution-state error.
- Use the reference feedback and gold error type as factual guidance, not as a text-overlap target.
- A response may use different wording from the reference and still receive a high score if it is diagnostically and pedagogically correct.

Return your evaluation strictly in JSON format only.

The JSON schema must be:
{{
  "diagnosis_accuracy_A": 1-5,
  "repair_correctness_A": 1-5,
  "bioinformatics_relevance_A": 1-5,
  "pedagogical_quality_A": 1-5,

  "diagnosis_accuracy_B": 1-5,
  "repair_correctness_B": 1-5,
  "bioinformatics_relevance_B": 1-5,
  "pedagogical_quality_B": 1-5,

  "diagnosis_accuracy_C": 1-5,
  "repair_correctness_C": 1-5,
  "bioinformatics_relevance_C": 1-5,
  "pedagogical_quality_C": 1-5,

  "best_response": "A" or "B" or "C" or "tie",
  "brief_reason": "one or two sentences explaining the evaluation"
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

def call_deepseek_judge(prompt):
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

            valid, msg = validate_judge_result(result)
            if not valid:
                raise ValueError(f"Invalid judge result: {msg}. Raw: {content}")

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
    print("DeepSeek LLM-as-a-Judge Feedback Quality Evaluation")
    print(f"Judge model: {JUDGE_MODEL}")
    print(f"Output file: {OUTPUT_FILE}")
    print("=" * 80)

    # Load all model predictions
    model_data = {}
    for model_name, file_path in MODEL_FILES.items():
        print(f"Loading {model_name}: {file_path}")
        model_data[model_name] = load_prediction_file(file_path)

    # Use common IDs only
    common_ids = None
    for model_name, id_to_item in model_data.items():
        ids = set(id_to_item.keys())
        common_ids = ids if common_ids is None else common_ids & ids

    common_ids = sorted(
        list(common_ids),
        key=lambda x: stable_int_hash(x)
    )

    if SAMPLE_LIMIT is not None:
        common_ids = common_ids[:SAMPLE_LIMIT]

    print(f"Common samples to evaluate: {len(common_ids)}")

    # Resume support
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

        # Use TaxSFT item as sample metadata source; all files share same input/gold
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

        prompt = build_judge_prompt(
            sample=sample,
            side_to_response=side_to_response
        )

        judge_result = call_deepseek_judge(prompt)

        if judge_result is None:
            results.append({
                "id": sample["id"],
                "status": "failed",
                "mapping": side_to_model
            })
        else:
            best_raw = judge_result.get("best_response", "tie")

            if best_raw in ["A", "B", "C"]:
                best_model = side_to_model[best_raw]
            else:
                best_model = "tie"

            results.append({
                "id": sample["id"],
                "status": "success",
                "judge_type": "score_based_three_model",
                "mapping": side_to_model,
                "best_response_raw": best_raw,
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