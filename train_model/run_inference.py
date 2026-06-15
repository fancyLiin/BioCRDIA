import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ==========================================
# 1. 配置区
# ==========================================
# 跑zero-shot修改MODEL_NAME = "Qwen2.5-7B-Instruct-Zero-shot"
# MODEL_NAME = "BioCRDIA-7B-RPD-0.3"
# MODEL_NAME = "Qwen2.5-7B-Instruct-Zero-shot"
MODEL_NAME = "BioCRDIA-7B-RPD-0.0"
BASE_MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"

# 如果是推理 LoRA 模型，填写 LoRA 路径
# 如果只推理 base model，把 LORA_PATH = None
# LORA_PATH = "/root/autodl-tmp/saves/BioCRDIA-7B-RPD-0.3"
LORA_PATH = "/root/autodl-tmp/saves/BioCRDIA-7B-RPD-0.0"
# LORA_PATH = None

TEST_DATA_PATH = "../train_data/biocrdia_test.json"

# OUTPUT_PREDICTION_PATH = "./result_inference/predictions_Qwen2.5-7B-Zeroshot.json"
# OUTPUT_PREDICTION_PATH = "./result_inference/predictions_BioCRDIA-7B-RPD-0.3.json"
OUTPUT_PREDICTION_PATH = "./result_inference/predictions_BioCRDIA-7B-RPD-0.0.json"

MAX_NEW_TOKENS = 1024


# ==========================================
# 2. Prompt 模板
# ==========================================

def format_prompt(instruction, user_input):
    return (
        f"<|im_start|>system\n"
        f"{instruction}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ==========================================
# 3. 加载模型
# ==========================================

def load_model_and_tokenizer():

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True
    )

    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    if LORA_PATH is not None:
        print(f"Loading LoRA adapter from: {LORA_PATH}")

        model = PeftModel.from_pretrained(
            model,
            LORA_PATH
        )

        print("LoRA adapter loaded.")

    else:
        print("No LoRA adapter. Running base model only.")

    model.eval()
    model.config.use_cache = True

    return model, tokenizer


# ==========================================
# 4. 单样本推理
# ==========================================

def generate_prediction(model, tokenizer, instruction, user_input):

    prompt = format_prompt(instruction, user_input)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model.device)

    input_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[0][input_length:]

    generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return generated_text.strip()


# ==========================================
# 5. 主推理流程
# ==========================================

def main():

    print("=" * 60)
    print(f"Inference Model: {MODEL_NAME}")
    print("=" * 60)

    model, tokenizer = load_model_and_tokenizer()

    print("Loading test dataset...")

    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    results = []

    print(f"Start inference on {len(test_data)} samples...")

    for idx, item in enumerate(tqdm(test_data)):

        prediction = generate_prediction(
            model=model,
            tokenizer=tokenizer,
            instruction=item["instruction"],
            user_input=item["input"]
        )

        results.append({
            "id": idx,
            "model_name": MODEL_NAME,
            "instruction": item["instruction"],
            "input": item["input"],
            "ground_truth": item["output"],
            "prediction": prediction
        })

    print(f"Saving predictions to: {OUTPUT_PREDICTION_PATH}")

    with open(OUTPUT_PREDICTION_PATH, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("Inference finished.")


if __name__ == "__main__":
    main()