import time
import torch
from datasets import load_dataset
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
import evaluate
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# List of models to benchmark
models_to_benchmark = ["t5-small", "t5-base"]

# Load the TREC dataset
dataset = load_dataset("trec")
label_names = dataset["train"].features["coarse_label"].names

# Preprocessing: convert example to text-to-text format
def preprocess(example):
    input_text = f"classify question: {example['text']}"
    target_text = label_names[example['coarse_label']]
    return {"input_text": input_text, "target_text": target_text}

dataset = dataset.map(preprocess)

# Tokenization function with dynamic padding
def tokenize(batch, tokenizer):
    model_inputs = tokenizer(
        batch["input_text"],
        padding=True,            # dynamic padding for inputs
        truncation=True,
        max_length=128           # cap max input length
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["target_text"],
            padding=True,        # dynamic padding for targets
            truncation=True      # in case label is long
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Load accuracy metric
metric = evaluate.load("accuracy")

# Function to compute accuracy during evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = tokenizer.batch_decode(torch.argmax(torch.tensor(logits), dim=-1), skip_special_tokens=True)
    references = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return metric.compute(predictions=predictions, references=references)

# Store benchmark results
results = []

# Loop through each model for training + evaluation
for model_name in models_to_benchmark:
    print(f"\n=== Running benchmark for {model_name} ===")

    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

    # Tokenize datasets with dynamic padding
    train_data = dataset["train"].map(lambda x: tokenize(x, tokenizer), batched=True)
    test_data = dataset["test"].map(lambda x: tokenize(x, tokenizer), batched=True)

    # Data collator handles dynamic padding at batch time
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Training configuration
    training_args = TrainingArguments(
        output_dir=f"./{model_name}-trec",
        evaluation_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="no",
        report_to="none"
    )

    # Trainer class wraps everything into a training loop
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train the model and measure training time
    start_time = time.time()
    trainer.train()
    elapsed_time = time.time() - start_time

    # Evaluate model on test set
    metrics = trainer.evaluate()
    results.append({
        "Model": model_name,
        "Accuracy": round(metrics["eval_accuracy"], 4),
        "Time (s)": round(elapsed_time, 2)
    })

# Display all benchmark results as a table
results_df = pd.DataFrame(results)
print("\n=== Benchmark Results ===")
print(results_df)
