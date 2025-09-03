# pip install -U "transformers>=4.53" "trl>=0.21.0" peft accelerate datasets

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import re


ds = load_dataset("jmcinern/Instruction_Ga_En_for_LoRA")


model_id = "jmcinern/qwen3-8b-base-cpt"
subfolder = "checkpoint-33000"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", trust_remote_code=True, subfolder=subfolder)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, subfolder=subfolder)


def to_messages(ex):
    user = ex["instruction"] + (("\n\n" + ex["context"]) if ex.get("context") else "")
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": ex["response"]},
    ]}

cols = ds["train"].column_names
ds = ds.map(to_messages, remove_columns=[c for c in cols if c != "messages"])


# Pre-render to plain text with thinking disabled
THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)

def to_text(ex):
    txt = tokenizer.apply_chat_template(
        ex["messages"], tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    if "<think>" in txt:
        txt = THINK_RE.sub("", txt)  # hard-disable thinking
    return {"text": txt}

ds = ds.map(to_text)

peft_cfg = LoraConfig(
    task_type="CAUSAL_LM",
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"],
)
model = get_peft_model(model, peft_cfg)
model.print_trainable_parameters()
model.config.use_cache = False  # because grad checkpointing


MAX_LEN = 4096
eval_split = "test" if "test" in ds else ("validation" if "validation" in ds else None)
eval_strategy = "steps" if eval_split else "no"
print("[INFO] Using eval split:", eval_split)
print("[INFO] Evaluation strategy:", eval_strategy)

sft_cfg = SFTConfig(
    output_dir="qwen3-8B-lora-ga",
    max_length=MAX_LEN,
    packing=True,                      # pack multiple samples up to max_length
    dataset_text_field="text",         # we pre-rendered text; do NOT pass messages here
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    num_train_epochs=3,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    max_steps=-1,
    logging_steps=20,
    eval_strategy = "steps",
    save_strategy="steps",
    save_steps=50,
    bf16=True,
    eval_steps=100,
    optim="adamw_torch",
    gradient_checkpointing=True,
    ddp_find_unused_parameters=False,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=sft_cfg,
    train_dataset=ds["train"],
    eval_dataset=ds[eval_split] if eval_split else None,
)

trainer.train()

# Final save (PEFT adapters only)
trainer.save_model()                 # saves to args.output_dir
tokenizer.save_pretrained(sft_cfg.output_dir)

print("[INFO] Saved artifacts in:", sft_cfg.output_dir)
print("[INFO] Files include adapter_model.safetensors, adapter_config.json, and tokenizer files.")
