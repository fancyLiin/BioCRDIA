#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
from tqdm import tqdm
from openai import OpenAI


# =====================================================
# 1. Configuration
# =====================================================

MODEL_NAME = "GPT-4o-Zeroshot-TaxTest"

# OpenAI API model
API_MODEL = "gpt-4o"

TEST_DATA_PATH = "../train_data_taxsft/biocrdia_tax_test.json"

OUTPUT_DIR = "./result_inference"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "predictions_GPT-4o_Zeroshot_TaxTest.json"
)

MAX_NEW_TOKENS = 1200
TEMPERATURE = 0

# API retry settings
MAX_RETRIES = 5
SLEEP_SECONDS = 0.2

# 每生成多少条保存一次，防止中途断掉
SAVE_EVERY = 1


# =====================================================
# 2. Taxonomy-aware zero-shot instruction
# =====================================================

ZERO_SHOT_INSTRUCTION = """You are BioCRDIA, a taxonomy-guided bioinformatics intelligent tutoring system.

Your task is to diagnose the student's error type and provide process-oriented pedagogical feedback.

You must choose exactly one error type from the following taxonomy:

1. NW_BOUNDARY_INITIALIZATION_ERROR
2. NW_GAP_PENALTY_SIGN_ERROR
3. NW_MISMATCH_SCORING_ERROR
4. PWM_LOG_ZERO_NO_PSEUDOCOUNT
5. PWM_PSEUDOCOUNT_DENOMINATOR_ERROR
6. PWM_BACKGROUND_PROBABILITY_ERROR
7. TRANSLATION_STOP_CODON_HANDLING_ERROR
8. TRANSLATION_READING_FRAME_ERROR
9. MUTATION_EFFECT_MISCLASSIFICATION

Your answer must strictly follow this format:

### 0. Diagnostic Label
Error Type: <one taxonomy label>
Evidence: <one concise evidence sentence>

### 1. Defect Analysis
<explain the root cause>

### 2. Repair Logic
<explain how to correct the reasoning>

### 3. Pedagogical Scaffolding
<give a reusable learning strategy>
"""


# =====================================================
# 3. Utilities
# =====================================================

def load_existing_results(output_file):
    """
    如果之前跑到一半中断了，可以从已有结果继续跑。
    """
    if not os.path.exists(output_file):
        return [], set()

    try:
        with open(output_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        done_ids = set(item["id"] for item in results if "id" in item)
        print(f"[Resume] Loaded {len(results)} existing results from {output_file}")
        return results, done_ids

    except Exception as e:
        print(f"[Warning] Failed to load existing output file: {e}")
        print("[Warning] Start from scratch.")
        return [], set()


def save_results(output_file, results):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def call_gpt4o(client, user_input):
    messages = [
        {
            "role": "system",
            "content": ZERO_SHOT_INSTRUCTION
        },
        {
            "role": "user",
            "content": user_input
        }
    ]

    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=API_MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_NEW_TOKENS,
            )

            prediction = response.choices[0].message.content

            if prediction is None:
                return ""

            return prediction.strip()

        except Exception as e:
            last_error = e
            wait_time = 2 ** attempt
            print(f"[Warning] API call failed: {e}")
            print(f"[Warning] Retry in {wait_time} seconds...")
            time.sleep(wait_time)

    raise RuntimeError(f"API call failed after {MAX_RETRIES} retries: {last_error}")


# =====================================================
# 4. Main Inference
# =====================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    api_key = ""

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Please run:\n"
            "export OPENAI_API_KEY='your_api_key'"
        )

    client = OpenAI(api_key=api_key)

    print("=" * 80)
    print("GPT-4o zero-shot inference on BioCRDIA Tax Test")
    print(f"API model:  {API_MODEL}")
    print(f"Test data:  {TEST_DATA_PATH}")
    print(f"Output:     {OUTPUT_FILE}")
    print("=" * 80)

    print("Loading test data...")

    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    results, done_ids = load_existing_results(OUTPUT_FILE)

    print(f"Total samples: {len(test_data)}")
    print(f"Already done:  {len(done_ids)}")
    print(f"Remaining:     {len(test_data) - len(done_ids)}")

    for idx, item in enumerate(tqdm(test_data)):

        item_id = item["id"]

        if item_id in done_ids:
            continue

        prediction = call_gpt4o(
            client=client,
            user_input=item["input"]
        )

        results.append({
            "id": item["id"],
            "model_name": MODEL_NAME,

            "instruction": ZERO_SHOT_INSTRUCTION,
            "input": item["input"],

            # 参考答案，用于后续 LLM-as-a-Judge / 文本质量评价
            "ground_truth": item.get("output", ""),

            # gold label，用于 Error Diagnosis Accuracy
            "error_type": item["error_type"],

            "domain": item.get("domain", ""),
            "sub_error_type": item.get("sub_error_type", ""),
            "reference_solution": item.get("reference_solution", ""),

            "prediction": prediction
        })

        done_ids.add(item_id)

        if len(results) % SAVE_EVERY == 0:
            save_results(OUTPUT_FILE, results)

        time.sleep(SLEEP_SECONDS)

    save_results(OUTPUT_FILE, results)

    print("=" * 80)
    print("GPT-4o zero-shot inference finished.")
    print(f"Saved predictions to: {OUTPUT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()