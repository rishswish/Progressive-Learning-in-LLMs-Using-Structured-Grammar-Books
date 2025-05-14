import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    pipeline,
)
import evaluate

# ==== Config ====
model_name = "gpt2"  # Try swapping with: "EleutherAI/gpt-neo-125M", "tiiuae/falcon-rw-1b"
output_dir = "./gpt-trec-output"
max_length = 64
batch_size = 8
num_epochs = 3
use_better_prompt = False
label_names = ['DESC', 'ENTY', 'ABBR', 'HUM', 'NUM', 'LOC']

# ==== Load Dataset ====
print("📦 Loading TREC dataset...")
dataset = load_dataset("trec", trust_remote_code=True)
train_data = dataset["train"]
test_data = dataset["test"]

# ==== Load Tokenizer & Model ====
print(f"🔍 Loading model and tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# ==== Format Prompts ====
def format_example(example):
    if use_better_prompt:
        prompt = f"Here is a question: {example['text']} What is the best label among {label_names}? Answer:"
    else:
        prompt = f"Classify this question: {example['text']}\nLabel:"
    label = label_names[example['coarse_label']]
    full_text = prompt + " " + label
    return tokenizer(full_text, truncation=True, padding="max_length", max_length=max_length)

print("🧹 Tokenizing datasets...")
train_dataset = train_data.map(format_example, remove_columns=train_data.column_names)
test_dataset = test_data.map(format_example, remove_columns=test_data.column_names)

# ==== Training Arguments ====
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    logging_dir=f"{output_dir}/logs",
    logging_steps=20,
    fp16=torch.cuda.is_available(),
    save_total_limit=2,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ==== Trainer ====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("🚀 Starting training...")
trainer.train()

# ==== Evaluation Using `evaluate` ====
print("📊 Evaluating model using Hugging Face `evaluate`...")
text_gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

def classify_question(text):
    if use_better_prompt:
        prompt = f"Here is a question: {text} What is the best label among {label_names}? Answer:"
    else:
        prompt = f"Classify this question: {text}\nLabel:"
    result = text_gen(prompt, max_new_tokens=3, do_sample=False)[0]["generated_text"]
    label = result.split("Answer:" if use_better_prompt else "Label:")[-1].strip().split()[0]
    return label.upper()

def clean_prediction(label):
    label = label.strip().upper().replace(".", "").replace(":", "")
    return label if label in label_names else "DESC"

y_true = [example["coarse_label"] for example in test_data]
y_pred_labels = [clean_prediction(classify_question(example["text"])) for example in test_data]
y_pred = [label_names.index(label) for label in y_pred_labels]

# ==== Evaluate with HF evaluate ====
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

acc_result = accuracy.compute(predictions=y_pred, references=y_true)
f1_result = f1.compute(predictions=y_pred, references=y_true, average="weighted")

print("\n✅ Evaluation Metrics (via evaluate):")
print(f"Accuracy: {acc_result['accuracy']:.4f}")
print(f"Weighted F1: {f1_result['f1']:.4f}")
