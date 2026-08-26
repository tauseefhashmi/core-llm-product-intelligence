"""Minimal QLoRA SFT template. Run on a CUDA machine/Colab with a suitable model."""
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
import torch

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATA = "training/train.jsonl"

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="auto")

data = load_dataset("json", data_files=DATA, split="train")

def fmt(x):
    return f"Instruction: {x['instruction']}\nInput: {x['input']}\nAnswer: {x['output']}"

trainer = SFTTrainer(
    model=model,
    train_dataset=data,
    formatting_func=fmt,
    peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM"),
    args=TrainingArguments(output_dir="./qlora-output", num_train_epochs=2, per_device_train_batch_size=2, logging_steps=1),
)
trainer.train()
trainer.save_model("./qlora-output")
