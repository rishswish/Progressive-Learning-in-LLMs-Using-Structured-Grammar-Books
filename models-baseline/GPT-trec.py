import time
import torch
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Local GPT-style models runnable on T4
models_to_train = [
    "gpt2",                      # 124M
    "EleutherAI/gpt-neo-125M",  # 125M
    "gpt2-medium",              # 355M
]

# 📦 Load dataset
dataset = load_dataset("trec")
label_names = dataset["train"].features["coarse_label"].names

# 🔧 Format input text + labels for GPT-style training
def preprocess(example):
    label = label_names[example["coarse_label"]]
    return {"full_text": f"classify question: {example['text']} {label}"}

dataset = dataset.map(preprocess)

# 🎯 Accuracy metric
metric = evaluate.load("accuracy")
results = []

# 🧪 Loop over models
for model_name in models_to_train:
    print(f"\n=== Training {model_name} ===")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # Ensure valid padding tokens
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Tokenize
    def tokenize(batch):
        tokens = tokenizer(
            batch["full_text"],
            padding=True,
            truncation=True,
            pad_to_multiple_of=8
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    train_dataset = dataset["train"].map(tokenize, batched=True)
    test_dataset = dataset["test"].map(tokenize, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    # Evaluation metric
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = torch.argmax(torch.tensor(logits), dim=-1)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        pred_labels = [p.strip().split()[-1] for p in decoded_preds]
        true_labels = [l.strip().split()[-1] for l in decoded_labels]
        return metric.compute(predictions=pred_labels, references=true_labels)

    # ⚙️ Training config
    training_args = TrainingArguments(
        output_dir=f"./{model_name.replace('/', '_')}_trec",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="no",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # ⏱️ Train and evaluate
    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time

    metrics = trainer.evaluate()
    results.append({
        "Model": model_name,
        "Accuracy": round(metrics["eval_accuracy"], 4),
        "Time (s)": round(elapsed, 2)
    })

# 📊 Show benchmark table
df = pd.DataFrame(results)
print("\n=== Benchmark Results ===")
print(df.to_markdown(index=False))
