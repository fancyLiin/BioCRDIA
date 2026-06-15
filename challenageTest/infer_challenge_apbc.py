#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import gc
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# =====================================================
# 1. Configuration
# =====================================================

BASE_MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"

TEST_DATA_PATH = "/root/autodl-tmp/APBC_vgpu/challenageTest/data/biocrdia_challenge90_hard.json"

SAVE_ROOT = "/root/autodl-tmp/saves"

OUTPUT_DIR = "/root/autodl-tmp/APBC_vgpu/challenageTest/result_data/challenge_infer_apbc_hard"

MAX_INPUT_LENGTH = 2048
MAX_NEW_TOKENS = 1200


# =====================================================
# 2. Instructions
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


EVAL_INSTRUCTION = """You are BioCRDIA, a taxonomy-guided bioinformatics intelligent tutoring system.

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
# 3. Models used in APBC paper
# =====================================================

MODEL_CONFIGS = [
    {
        "model_name": "Qwen2.5-7B-ZeroShot",
        "model_type": "base",
        "lora_path": None,
        "instruction_mode": "zero_shot",
    },
    {
        "model_name": "BioCRDIA-7B-VanillaSFT-TaxData",
        "model_type": "lora",
        "lora_path": os.path.join(SAVE_ROOT, "BioCRDIA-7B-VanillaSFT-TaxData"),
        "instruction_mode": "eval",
    },
    {
        "model_name": "BioCRDIA-7B-TaxSFT-NeutralInstr",
        "model_type": "lora",
        "lora_path": os.path.join(SAVE_ROOT, "BioCRDIA-7B-TaxSFT-NeutralInstr"),
        "instruction_mode": "eval",
    },
    {
        "model_name": "BioCRDIA-7B-TaxSFT",
        "model_type": "lora",
        "lora_path": os.path.join(SAVE_ROOT, "BioCRDIA-7B-TaxSFT"),
        "instruction_mode": "eval",
    }
]


# =====================================================
# 4. Prompt Builder
# =====================================================

def format_prompt(instruction, user_input):
    """
    Must be consistent with the original training / inference ChatML format.
    """
    prompt = (
        f"<|im_start|>system\n"
        f"{instruction}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt


def select_instruction(item, instruction_mode):
    """
    instruction_mode:
    - zero_shot: use ZERO_SHOT_INSTRUCTION
    - eval: use EVAL_INSTRUCTION
    - dataset: use item["instruction"], matching your TaxSFT inference script
    """
    if instruction_mode == "zero_shot":
        return ZERO_SHOT_INSTRUCTION

    if instruction_mode == "eval":
        return EVAL_INSTRUCTION

    if instruction_mode == "dataset":
        return item["instruction"]

    raise ValueError(f"Unknown instruction_mode: {instruction_mode}")


# =====================================================
# 5. Text Cleaning
# =====================================================

def clean_generation(text):
    stop_tokens = [
        "<|im_end|>",
        "<|endoftext|>"
    ]

    for token in stop_tokens:
        if token in text:
            text = text.split(token)[0]

    return text.strip()


# =====================================================
# 6. Loading utilities
# =====================================================

def get_torch_dtype():
    """
    V100 usually does not support bf16, so fp16 is safer.
    If your previous environment supports bf16, this will automatically use bf16.
    """
    if not torch.cuda.is_available():
        return torch.float32

    if torch.cuda.is_bf16_supported():
        return torch.bfloat16

    return torch.float16


def load_tokenizer(model_config):
    """
    Keep the same logic as your TaxSFT inference script:
    try LoRA path tokenizer first, then base tokenizer.
    """
    lora_path = model_config.get("lora_path")

    if lora_path is not None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                lora_path,
                trust_remote_code=True
            )
            return tokenizer
        except Exception:
            pass

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True
    )
    return tokenizer


def load_model(model_config):
    model_name = model_config["model_name"]
    model_type = model_config["model_type"]
    lora_path = model_config.get("lora_path")

    dtype = get_torch_dtype()

    print("=" * 80)
    print(f"Loading model: {model_name}")
    print(f"Base model: {BASE_MODEL_PATH}")
    print(f"LoRA path:  {lora_path}")
    print(f"Model type: {model_type}")
    print(f"Dtype:      {dtype}")
    print("=" * 80)

    tokenizer = load_tokenizer(model_config)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True
    )

    if model_type == "base":
        model = base_model

    elif model_type == "lora":
        if lora_path is None:
            raise ValueError(f"LoRA model {model_name} has no lora_path.")

        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA path not found: {lora_path}")

        model = PeftModel.from_pretrained(
            base_model,
            lora_path
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.eval()

    eos_token_ids = []

    if tokenizer.eos_token_id is not None:
        eos_token_ids.append(tokenizer.eos_token_id)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id >= 0:
        eos_token_ids.append(im_end_id)

    eos_token_ids = list(set(eos_token_ids))

    return model, tokenizer, eos_token_ids


# =====================================================
# 7. Data loading
# =====================================================

def load_test_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test data path not found: {path}")

    if path.endswith(".jsonl"):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict) and "data" in data:
        return data["data"]

    raise ValueError("Unsupported test data format. Please use JSON list or JSONL.")


# =====================================================
# 8. Single-model inference
# =====================================================

def run_inference_for_one_model(model_config, test_data):
    model_name = model_config["model_name"]
    instruction_mode = model_config["instruction_mode"]

    output_file = os.path.join(
        OUTPUT_DIR,
        f"predictions_{model_name}_challenge.json"
    )

    print("=" * 80)
    print(f"Running challenge inference for: {model_name}")
    print(f"Instruction mode: {instruction_mode}")
    print(f"Output file: {output_file}")
    print("=" * 80)

    model, tokenizer, eos_token_ids = load_model(model_config)

    results = []

    for item in tqdm(test_data, desc=model_name):
        instruction = select_instruction(item, instruction_mode)
        user_input = item["input"]

        prompt = format_prompt(
            instruction=instruction,
            user_input=user_input
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_LENGTH
        )

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
            "id": item.get("id", ""),
            "model_name": model_name,
            "instruction_mode": instruction_mode,

            "instruction": instruction,
            "input": item.get("input", ""),

            # challenge set usually has empty output; keep this field for compatibility
            "ground_truth": item.get("output", ""),

            # gold label for Error Diagnosis Accuracy
            "error_type": item.get("error_type", ""),

            # metadata
            "domain": item.get("domain", ""),
            "sub_error_type": item.get("sub_error_type", ""),
            "reference_solution": item.get("reference_solution", ""),
            "challenge_type": item.get("challenge_type", ""),

            # model output
            "prediction": prediction
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(f"Inference finished for {model_name}.")
    print(f"Saved predictions to: {output_file}")
    print("=" * 80)

    del model
    del tokenizer
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =====================================================
# 9. Main
# =====================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("APBC Template-disjoint Challenge Inference")
    print(f"Test data: {TEST_DATA_PATH}")
    print(f"Output dir: {OUTPUT_DIR}")
    print("=" * 80)

    test_data = load_test_data(TEST_DATA_PATH)

    print(f"Loaded {len(test_data)} challenge samples.")

    for model_config in MODEL_CONFIGS:
        run_inference_for_one_model(model_config, test_data)

    print("=" * 80)
    print("All challenge inference finished.")
    print("=" * 80)


if __name__ == "__main__":
    main()