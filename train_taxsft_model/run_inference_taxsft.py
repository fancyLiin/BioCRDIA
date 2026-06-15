#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# =====================================================
# 1. Configuration
# =====================================================

MODEL_NAME = "BioCRDIA-7B-TaxSFT"

BASE_MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
LORA_PATH = "/root/autodl-tmp/saves/BioCRDIA-7B-TaxSFT"

TEST_DATA_PATH = "../train_data_taxsft/biocrdia_tax_test.json"

OUTPUT_DIR = "./result_inference"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "predictions_BioCRDIA-7B-TaxSFT.json"
)

MAX_INPUT_LENGTH = 2048
MAX_NEW_TOKENS = 1200


# =====================================================
# 2. Prompt Builder
# =====================================================

def format_prompt(instruction, user_input):
    """
    Must be consistent with the training ChatML format.
    """

    prompt = (
        f"<|im_start|>system\n"
        f"{instruction}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    return prompt


# =====================================================
# 3. Text Cleaning
# =====================================================

def clean_generation(text):
    """
    Remove possible trailing ChatML tokens.
    """

    stop_tokens = [
        "<|im_end|>",
        "<|endoftext|>"
    ]

    for token in stop_tokens:
        if token in text:
            text = text.split(token)[0]

    return text.strip()


# =====================================================
# 4. Main Inference
# =====================================================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("BioCRDIA TaxSFT Inference")
    print(f"Model name: {MODEL_NAME}")
    print(f"Base model: {BASE_MODEL_PATH}")
    print(f"LoRA path:  {LORA_PATH}")
    print(f"Test data:  {TEST_DATA_PATH}")
    print(f"Output:     {OUTPUT_FILE}")
    print("=" * 80)

    print("Loading tokenizer...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            LORA_PATH,
            trust_remote_code=True
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_PATH,
            trust_remote_code=True
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    print("Loading base model...")

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    print("Loading LoRA adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        LORA_PATH
    )

    model.eval()

    # eos 设置：同时兼容 Qwen 的 <|im_end|>
    eos_token_ids = []

    if tokenizer.eos_token_id is not None:
        eos_token_ids.append(tokenizer.eos_token_id)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    if isinstance(im_end_id, int) and im_end_id >= 0:
        eos_token_ids.append(im_end_id)

    eos_token_ids = list(set(eos_token_ids))

    print("Loading test data...")

    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    results = []

    print(f"Running inference on {len(test_data)} samples...")

    for item in tqdm(test_data):

        prompt = format_prompt(
            instruction=item["instruction"],
            user_input=item["input"]
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_LENGTH
        )

        # 单卡或 device_map=auto 下通常放到模型首个 device 即可
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                eos_token_id=eos_token_ids if eos_token_ids else tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

        generated_tokens = outputs[0][input_length:]
        prediction = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=False
        )

        prediction = clean_generation(prediction)

        results.append({
            "id": item["id"],
            "model_name": MODEL_NAME,

            "instruction": item["instruction"],
            "input": item["input"],

            # 用于后续 LLM-as-a-Judge / 文本质量评估
            "ground_truth": item["output"],

            # 用于后续 Error Diagnosis Accuracy
            "error_type": item["error_type"],

            # 可选元数据，方便按领域/类型统计
            "domain": item.get("domain", ""),
            "sub_error_type": item.get("sub_error_type", ""),
            "reference_solution": item.get("reference_solution", ""),

            # 模型生成结果
            "prediction": prediction
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("Inference finished.")
    print(f"Saved predictions to: {OUTPUT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()