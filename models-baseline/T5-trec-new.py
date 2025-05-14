import time
import torch
from datasets import load_dataset
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq, LogitsProcessorList
)
import evaluate
import pandas as pd
from sklearn.metrics import f1_score, recall_score
from transformers import LogitsProcessor

class RestrictToLabelTokensProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = set(allowed_token_ids)

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float('-inf'))
        for token_id in self.allowed_token_ids:
            mask[:, token_id] = 0
        return scores + mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_to_benchmark = ["t5-small", "t5-base"]

dataset = load_dataset("trec")
label_names = dataset["train"].features["coarse_label"].names
label2id = {label: i for i, label in enumerate(label_names)}
id2label = {i: label for label, i in label2id.items()}

def preprocess(example):
    input_text = f"{example['text']}"
    target_text = label_names[example['coarse_label']]
    return {"input_text": input_text, "target_text": target_text}

dataset = dataset.map(preprocess)

def tokenize(batch, tokenizer):
    model_inputs = tokenizer(batch["input_text"], padding=True, truncation=True, max_length=128)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch["target_text"], padding=True, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

accuracy_metric = evaluate.load("accuracy")

results = []

for model_name in models_to_benchmark:
    print(f"\n=== Benchmarking {model_name} ===")

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

    allowed_token_ids = tokenizer.convert_tokens_to_ids(label_names)
    logits_processor = LogitsProcessorList([
        RestrictToLabelTokensProcessor(allowed_token_ids)
    ])

    train_data = dataset["train"].map(lambda x: tokenize(x, tokenizer), batched=True)
    test_data = dataset["test"].map(lambda x: tokenize(x, tokenizer), batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=f"./{model_name}-trec",
        evaluation_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="no",
        report_to="none"
    )

    predictions_to_save = {"input": [], "prediction": [], "target": []}

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        if not isinstance(logits, torch.Tensor):
            logits = torch.tensor(logits)

        pred_ids = torch.argmax(logits, dim=-1)
        decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        pred_labels = [label2id.get(p.strip().split()[-1], -1) for p in decoded_preds]
        true_labels = [label2id.get(l.strip().split()[-1], -1) for l in decoded_labels]

        filtered = [(p, t, inp, pred, tgt) for p, t, inp, pred, tgt in zip(
            pred_labels, true_labels,
            dataset["test"]["input_text"],
            decoded_preds,
            decoded_labels
        )]
        
        if filtered:
            pred_labels, true_labels, inputs, preds, targets = zip(*filtered)
        else:
            return {}

        predictions_to_save["input"].extend(inputs)
        predictions_to_save["prediction"].extend(preds)
        predictions_to_save["target"].extend(targets)

        return {
            "accuracy": accuracy_metric.compute(predictions=pred_labels, references=true_labels)["accuracy"],
            "f1": f1_score(true_labels, pred_labels, average="macro"),
            "recall": recall_score(true_labels, pred_labels, average="macro")
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # === Separate timing ===
    start_train = time.time()
    trainer.train()
    end_train = time.time()

    start_eval = time.time()
    metrics = trainer.evaluate()
    end_eval = time.time()

    train_time = end_train - start_train
    eval_time = end_eval - start_eval

    # Save predictions
    df_preds = pd.DataFrame(predictions_to_save)
    df_preds.to_csv(f"{model_name}_predictions.csv", index=False)

    results.append({
        "Model": model_name,
        "Accuracy": round(metrics.get("eval_accuracy", 0), 4),
        "F1": round(metrics.get("eval_f1", 0), 4),
        "Recall": round(metrics.get("eval_recall", 0), 4),
        "Train Time (s)": round(train_time, 2),
        "Eval Time (s)": round(eval_time, 2)
    })

results_df = pd.DataFrame(results)
print("\n=== Benchmark Results ===")
print(results_df)
