import os
import json
import time
import random
from tqdm import tqdm
from openai import OpenAI


# =====================================================
# 1. Configuration
# =====================================================

DEEPSEEK_API_KEY = "

# DeepSeek official OpenAI-compatible endpoint
BASE_URL = "https://api.deepseek.com"

# 可根据你的账号可用模型调整
JUDGE_MODEL = "deepseek-chat"

ZERO_SHOT_FILE = "./result_inference/predictions_Qwen2.5-7B-Zeroshot.json"
RPD_FILE = "./result_inference/predictions_BioCRDIA-7B-RPD-0.3.json"

OUTPUT_FILE = "./result_data/judge_pairwise_deepseek_rpd03_vs_zeroshot.json"

SEED = 42
MAX_RETRY = 3
SLEEP_TIME = 1.0


random.seed(SEED)


# =====================================================
# 2. DeepSeek Client
# =====================================================

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=BASE_URL
)


# =====================================================
# 3. Prompt Builder
# =====================================================

def build_judge_prompt(sample, response_a, response_b):
    return f"""
You are an expert evaluator for bioinformatics intelligent tutoring systems.

Your task is to compare two model responses to the same student problem.
Evaluate which response provides better process-oriented pedagogical feedback.

You must judge based on the following four dimensions:

1. Diagnosis Accuracy:
Whether the response correctly identifies the root cause of the student's error.

2. Repair Correctness:
Whether the response provides a correct and actionable repair strategy.

3. Bioinformatics Relevance:
Whether the response correctly uses relevant bioinformatics concepts.

4. Pedagogical Quality:
Whether the response provides useful learning guidance, conceptual explanation, and scaffolding.

Important rules:
- Do not prefer a response just because it is longer.
- Do not prefer a response just because it sounds fluent.
- Penalize hallucinated biological or computational claims.
- Penalize responses that miss the actual execution-state error.
- If both responses are similarly good, choose "tie".
- Use the reference answer only as guidance, not as a text-overlap target.

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
  "winner": "A" or "B" or "tie",
  "brief_reason": "one or two sentences"
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
# 4. API Call
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
            return json.loads(content)

        except Exception as e:
            print(f"[Retry {attempt + 1}/{MAX_RETRY}] Error: {e}")
            time.sleep(SLEEP_TIME * (attempt + 1))

    return None


# =====================================================
# 5. Main Evaluation
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

    for i in tqdm(range(len(rpd_data))):

        zero_item = zero_data[i]
        rpd_item = rpd_data[i]

        assert zero_item["id"] == rpd_item["id"], f"ID mismatch at index {i}"

        sample = {
            "id": rpd_item["id"],
            "input": rpd_item["input"],
            "ground_truth": rpd_item["ground_truth"]
        }

        zero_pred = zero_item["prediction"]
        rpd_pred = rpd_item["prediction"]

        # Blind pairwise setting: randomly assign A/B
        if random.random() < 0.5:
            response_a = zero_pred
            response_b = rpd_pred
            mapping = {
                "A": "zero-shot",
                "B": "BioCRDIA-7B-RPD-0.3"
            }
        else:
            response_a = rpd_pred
            response_b = zero_pred
            mapping = {
                "A": "BioCRDIA-7B-RPD-0.3",
                "B": "zero-shot"
            }

        prompt = build_judge_prompt(
            sample=sample,
            response_a=response_a,
            response_b=response_b
        )

        judge_result = call_deepseek_judge(prompt)

        if judge_result is None:
            results.append({
                "id": sample["id"],
                "status": "failed"
            })
            continue

        winner_raw = judge_result.get("winner", "tie")

        if winner_raw in ["A", "B"]:
            winner_model = mapping[winner_raw]
        else:
            winner_model = "tie"

        result = {
            "id": sample["id"],
            "status": "success",
            "mapping": mapping,
            "winner_raw": winner_raw,
            "winner_model": winner_model,
            "scores": judge_result,
        }

        results.append(result)

        # 每条都保存，防止中途断掉全部丢失
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        time.sleep(SLEEP_TIME)

    print(f"Saved judge results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()