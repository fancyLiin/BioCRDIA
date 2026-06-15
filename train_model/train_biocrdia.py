import random
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed
)

from peft import (
    LoraConfig,
    get_peft_model
)

from datasets import load_dataset


# =====================================================
# Configuration
# =====================================================

RPD_PROBABILITY = 0.0

SEED = 42

MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"

TRAIN_DATA_PATH = "../train_data/biocrdia_train.json"
VAL_DATA_PATH = "../train_data/biocrdia_val.json"

OUTPUT_DIR = (
    f"/root/autodl-tmp/saves/"
    f"BioCRDIA-7B-RPD-{RPD_PROBABILITY}"
)

MAX_LENGTH = 2048

set_seed(SEED)
random.seed(SEED)


# =====================================================
# Prompt Builder
# =====================================================

def format_prompt(instruction, user_input, output=None):

    prompt = (
        f"<|im_start|>system\n"
        f"{instruction}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    if output is not None:
        prompt += output

    return prompt


# =====================================================
# RPD Mechanism
# =====================================================

def apply_rpd(user_input):

    if "[Student Execution State]" not in user_input:
        return user_input

    parts = user_input.split("[Student Execution State]")

    return (
        parts[0]
        + "[Student Execution State]\n"
        + "[Execution State Omitted by RPD. "
          "Infer the defect purely from the algorithmic context.]"
    )


# =====================================================
# Dataset Preprocessing
# =====================================================

def preprocess_function(examples, tokenizer, is_training=True):

    input_texts = []
    prompt_lengths = []

    for i in range(len(examples["instruction"])):

        instruction = examples["instruction"][i]
        user_input = examples["input"][i]
        output = examples["output"][i]

        if is_training and random.random() < RPD_PROBABILITY:
            user_input = apply_rpd(user_input)

        full_text = format_prompt(
            instruction,
            user_input,
            output
        )

        prompt_only = format_prompt(
            instruction,
            user_input
        )

        prompt_len = len(
            tokenizer(
                prompt_only,
                add_special_tokens=False
            )["input_ids"]
        )

        input_texts.append(full_text)
        prompt_lengths.append(prompt_len)

    model_inputs = tokenizer(
        input_texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False
    )

    labels = []

    for ids, prompt_len in zip(
            model_inputs["input_ids"],
            prompt_lengths):

        prompt_len = min(prompt_len, len(ids))

        label = (
            [-100] * prompt_len
            + ids[prompt_len:]
        )

        labels.append(label)

    model_inputs["labels"] = labels

    return model_inputs


# =====================================================
# Main
# =====================================================

def main():

    print("=" * 60)
    print(f"BioCRDIA Training")
    print(f"RPD = {RPD_PROBABILITY}")
    print(f"Seed = {SEED}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )

    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        print("Flash Attention Enabled")
    except:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        print("Flash Attention Not Available")

    model.gradient_checkpointing_enable()

    print("Configuring LoRA...")

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ]
    )

    model = get_peft_model(
        model,
        lora_config
    )

    model.print_trainable_parameters()

    print("Loading dataset...")

    dataset = load_dataset(
        "json",
        data_files={
            "train": TRAIN_DATA_PATH,
            "val": VAL_DATA_PATH
        }
    )

    train_dataset = dataset["train"].map(
        lambda x: preprocess_function(
            x,
            tokenizer,
            is_training=True
        ),
        batched=True,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False
    )

    val_dataset = dataset["val"].map(
        lambda x: preprocess_function(
            x,
            tokenizer,
            is_training=False
        ),
        batched=True,
        remove_columns=dataset["val"].column_names,
        load_from_cache_file=False
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        learning_rate=5e-5,

        num_train_epochs=5,

        per_device_train_batch_size=4,

        gradient_accumulation_steps=4,

        per_device_eval_batch_size=4,

        warmup_ratio=0.1,

        weight_decay=0.01,

        lr_scheduler_type="cosine",

        bf16=True,

        logging_steps=10,

        eval_strategy="epoch",

        save_strategy="epoch",

        save_total_limit=1,

        load_best_model_at_end=True,

        metric_for_best_model="eval_loss",

        greater_is_better=False,

        seed=SEED,

        data_seed=SEED,

        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    print("Starting training...")

    trainer.train()

    print("Saving model...")

    trainer.save_model(OUTPUT_DIR)

    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Training Finished")


if __name__ == "__main__":
    main()