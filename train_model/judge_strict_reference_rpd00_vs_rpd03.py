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

# 这里改成你的两个推理结果文件
MODEL_A_FILE = "./result_inference/predictions_BioCRDIA-7B-RPD-0.0.json"
MODEL_B_FILE = "./result_inference/predictions_BioCRDIA-7B-RPD-0.3.json"

# 输出文件
OUTPUT_FILE = "./result_data/judge_strict_reference_rpd00_vs_rpd03.json"

# 模型名称
MODEL_A_NAME = "BioCRDIA-7B-RPD-0.0"
MODEL_B_NAME = "BioCRDIA-7B-RPD-0.3"

# 统计时重点看 p=0.3 是否优于 p=0.0
BASELINE_MODEL = MODEL_A_NAME
TARGET_MODEL = MODEL_B_NAME

COMPARE_TAG = "strict_reference_rpd0_vs_rpd03"

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
    raise KeyError("No prediction field found. Expected 'prediction' or 'model_prediction'.")


def assign_blind_pair(pred_a, pred_b, sample_id):
    """
    固定随机盲评：
    MODEL_A = RPD-0.1
    MODEL_B = RPD-0.3
    DeepSeek 只看到 Response A / Response B，不知道模型名。
    """
    rng = random.Random(SEED + int(sample_id))

    if rng.random() < 0.5:
        response_a = pred_a
        response_b = pred_b
        mapping = {
            "A": MODEL_A_NAME,
            "B": MODEL_B_NAME
        }
    else:
        response_a = pred_b
        response_b = pred_a
        mapping = {
            "A": MODEL_B_NAME,
            "B": MODEL_A_NAME
        }

    return response_a, response_b, mapping


# =====================================================
# 3. Strict-reference Judge Prompt
# =====================================================

def build_strict_reference_prompt(sample, response_a, response_b):
    return f"""
You are a strict expert evaluator for bioinformatics intelligent tutoring systems.

Your task is to compare two model responses to the same student problem.
Evaluate which response provides better process-oriented pedagogical feedback.

You must judge based on the following four dimensions:

1. Diagnosis Accuracy:
Whether the response correctly identifies the root cause of the student's error based on the execution state.

2. Repair Correctness:
Whether the response provides a correct, actionable, and case-specific repair strategy.

3. Bioinformatics Relevance:
Whether the response correctly uses relevant bioinformatics concepts, algorithms, and biological interpretation.

4. Pedagogical Quality:
Whether the response provides useful learning guidance, conceptual explanation, and scaffolding.

Strict evaluation rules:
- Do not prefer a response just because it is longer.
- Do not prefer a response just because it uses a cleaner structure.
- Do not prefer a response merely because it resembles the reference answer in wording.
- Penalize hallucinated biological, computational, or algorithmic claims.
- Penalize responses that miss the actual execution-state error.
- Penalize any incorrect numerical claim, especially incorrect alignment scores, probabilities, protein sequences, matrix values, or final scores.
- Penalize a response if it identifies the general error type but gives an incorrect case-specific explanation.
- Penalize a response if it contradicts a key fact in the reference answer.
- A response should not receive Diagnosis Accuracy above 4 if it contradicts any key factual detail in the reference answer.
- A response should not receive Repair Correctness above 4 if the proposed fix is incomplete or contains incorrect numerical consequences.
- If both responses are similarly good, choose "tie".
- Use the reference answer as factual guidance, not as a text-overlap target.

The winner should be based primarily on factual correctness and execution-state diagnosis, not formatting similarity.

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

[Reference Answer]
{sample["ground_truth"]}

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
                        "content": "You are a strict academic evaluator. Output valid JSON only."
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

    with open(MODEL_A_FILE, "r", encoding="utf-8") as f:
        model_a_data = json.load(f)

    with open(MODEL_B_FILE, "r", encoding="utf-8") as f:
        model_b_data = json.load(f)

    assert len(model_a_data) == len(model_b_data), "Two prediction files have different lengths."

    results = []

    # 支持断点续跑
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            results = json.load(f)

    finished_ids = {
        item["id"]
        for item in results
        if item.get("status") == "success"
        and item.get("compare_tag") == COMPARE_TAG
    }

    print("=" * 60)
    print("Strict-reference LLM-as-a-Judge")
    print(f"Compare: {MODEL_A_NAME} vs {MODEL_B_NAME}")
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 60)

    for i in tqdm(range(len(model_b_data))):

        model_a_item = model_a_data[i]
        model_b_item = model_b_data[i]

        assert model_a_item["id"] == model_b_item["id"], f"ID mismatch at index {i}"

        sample_id = model_b_item["id"]

        if sample_id in finished_ids:
            continue

        # 额外检查，避免两个文件不是同一批测试集
        assert model_a_item["input"] == model_b_item["input"], f"Input mismatch at id {sample_id}"
        assert model_a_item["ground_truth"] == model_b_item["ground_truth"], f"Ground truth mismatch at id {sample_id}"

        sample = {
            "id": sample_id,
            "input": model_b_item["input"],
            "ground_truth": model_b_item["ground_truth"]
        }

        pred_a = get_prediction(model_a_item)  # RPD-0.1
        pred_b = get_prediction(model_b_item)  # RPD-0.3

        response_a, response_b, mapping = assign_blind_pair(
            pred_a=pred_a,
            pred_b=pred_b,
            sample_id=sample_id
        )

        prompt = build_strict_reference_prompt(
            sample=sample,
            response_a=response_a,
            response_b=response_b
        )

        judge_result = call_deepseek_judge(prompt)

        if judge_result is None:
            results.append({
                "id": sample_id,
                "status": "failed",
                "compare_tag": COMPARE_TAG,
                "judge_type": "strict_reference",
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
                "compare_tag": COMPARE_TAG,
                "judge_type": "strict_reference",
                "model_a": MODEL_A_NAME,
                "model_b": MODEL_B_NAME,
                "mapping": mapping,
                "winner_raw": winner_raw,
                "winner_model": winner_model,
                "scores": judge_result
            })

        # 每条保存一次，防止中途断掉
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        time.sleep(SLEEP_TIME)

    print(f"Saved strict-reference judge results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()