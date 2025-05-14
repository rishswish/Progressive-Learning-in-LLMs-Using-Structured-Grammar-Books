import gc, torch, pandas as pd, time
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer,
    TrainingArguments, DataCollatorForLanguageModeling, pipeline
)
import evaluate
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.preprocessing import label_binarize

# ── Config ─────────────────────────────────────────────────────────────────────
models_to_train = [
    "gpt2",
    "EleutherAI/gpt-neo-125M",
    "gpt2-medium",
]

output_dir_root   = "./gpt-trec-output"
max_length        = 64
batch_size        = 8
num_epochs        = 3
use_better_prompt = False
label_names       = ['DESC', 'ENTY', 'ABBR', 'HUM', 'NUM', 'LOC', 'UNKNOWN']

# ── Data ───────────────────────────────────────────────────────────────────────
print("\U0001F4E6 Loading TREC dataset...")
dataset    = load_dataset("trec", trust_remote_code=True)
train_data = dataset["train"]
test_data  = dataset["test"]

# ── Metrics ─────────────────────────────────────────────────────────────────────
f1_metric        = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric    = evaluate.load("recall")
accuracy_metric  = evaluate.load("accuracy")

# ── Helpers ─────────────────────────────────────────────────────────────────────
def format_example(example, tok):
    if use_better_prompt:
        prompt = f"Here is a question: {example['text']} What is the best label among {label_names[:-1]}? Answer:"
    else:
        prompt = f"Classify this question: {example['text']}\nLabel:"
    label = label_names[example['coarse_label']]
    return tok(prompt + " " + label,
               truncation=True, padding="max_length", max_length=max_length)

def clean_prediction(label):
    label = label.strip().upper().replace(".", "").replace(":", "")
    return label if label in label_names else "UNKNOWN"

# ── Train‑and‑evaluate loop ────────────────────────────────────────────────────
def train_and_evaluate(model_name: str) -> dict:
    print(f"\n\U0001F50D Loading model & tokenizer: {model_name}")
    tok    = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    model  = AutoModelForCausalLM.from_pretrained(model_name)

    train_ds = train_data.map(lambda ex: format_example(ex, tok),
                              remove_columns=train_data.column_names)
    test_ds  = test_data.map(lambda ex: format_example(ex, tok),
                             remove_columns=test_data.column_names)

    args = TrainingArguments(
        output_dir=f"{output_dir_root}/{model_name.split('/')[-1]}",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        logging_dir=f"{output_dir_root}/{model_name.split('/')[-1]}/logs",
        logging_steps=20,
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tok,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    )

    print("\U0001F680 Training…")
    start_train = time.time()
    trainer.train()
    end_train = time.time()
    train_time = end_train - start_train
    print(f"⏱️ Training time: {train_time:.2f} seconds")

    # ── Inference ───────────────────────────────────────────────────────────────
    gen = pipeline("text-generation", model=model, tokenizer=tok,
                   device=0 if torch.cuda.is_available() else -1)

    def classify(text: str) -> str:
        if use_better_prompt:
            prompt = (f"Here is a question: {text} "
                      f"What is the best label among {label_names[:-1]}? Answer:")
        else:
            prompt = f"Classify this question: {text}\nLabel:"

        out = gen(prompt, max_new_tokens=3, do_sample=False)[0]["generated_text"]
        completion = out[len(prompt):].strip()
        if not completion:
            return "UNKNOWN"
        label = completion.split()[0]
        return clean_prediction(label)

    print("\U0001F50D Running inference on test set...")
    start_infer = time.time()
    y_true = [ex["coarse_label"] for ex in test_data]
    y_pred_labels = [classify(ex["text"]) for ex in test_data]
    y_pred = [label_names.index(lbl) for lbl in y_pred_labels]
    end_infer = time.time()
    infer_time = end_infer - start_infer
    unknown_count = y_pred_labels.count("UNKNOWN")

    print(f"⏱️ Inference time: {infer_time:.2f} seconds")
    print(f"⚠️  {unknown_count} UNKNOWN predictions out of {len(test_data)}")

    acc = accuracy_metric.compute(predictions=y_pred, references=y_true)["accuracy"]
    f1_weighted        = f1_metric.compute(predictions=y_pred, references=y_true, average="weighted")["f1"]
    f1_macro           = f1_metric.compute(predictions=y_pred, references=y_true, average="macro")["f1"]
    precision_weighted = precision_metric.compute(predictions=y_pred, references=y_true, average="weighted")["precision"]
    recall_weighted    = recall_metric.compute(predictions=y_pred, references=y_true, average="weighted")["recall"]
    precision_macro    = precision_metric.compute(predictions=y_pred, references=y_true, average="macro")["precision"]
    recall_macro       = recall_metric.compute(predictions=y_pred, references=y_true, average="macro")["recall"]

    # ── AUROC (macro) ──────────────────────────────────────────────────────────
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(len(label_names))))
        y_pred_bin = label_binarize(y_pred, classes=list(range(len(label_names))))

        roc_auc = roc_auc_score(y_true_bin, y_pred_bin, average="macro", multi_class="ovr")
        RocCurveDisplay.from_predictions(
            y_true=y_true_bin.argmax(axis=1),
            y_pred=y_pred_bin,
            name=f"{model_name} (AUROC = {roc_auc:.2f})",
            plot_chance_level=True,
        )
        plt.title(f"AUROC Curve - {model_name}")
        plt.savefig(f"{model_name.replace('/', '_')}_auroc.png")
        plt.close()
    except Exception as e:
        print(f"⚠️ Could not compute AUROC for {model_name}: {e}")
        roc_auc = None

    del model, trainer, tok, gen
    torch.cuda.empty_cache(); gc.collect()

    return {
        "model": model_name,
        "accuracy": acc,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "train_time_sec": round(train_time, 2),
        "inference_time_sec": round(infer_time, 2),
        "unknown_count": unknown_count,
        "auroc_macro": round(roc_auc, 4) if roc_auc is not None else "N/A",
    }

# ── Run all models ─────────────────────────────────────────────────────────────
results = [train_and_evaluate(m) for m in models_to_train]

# ── Save & show table ──────────────────────────────────────────────────────────
df = pd.DataFrame(results)
df.to_csv("results.csv", index=False)
df[["model", "auroc_macro"]].to_csv("auroc_scores.csv", index=False)

print("\n✅ Final results")
print(df.to_string(index=False))
