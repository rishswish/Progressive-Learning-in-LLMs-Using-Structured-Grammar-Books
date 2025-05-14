import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import evaluate
from transformers import TextGenerationPipeline
from tqdm import tqdm

# ==== Config ====
model_name = "gpt2"  
output_dir = "./gpt-jfleg-output"
max_length = 128
batch_size = 4
num_epochs = 3

# ==== Load Dataset ====
print("Loading JFLEG dataset...")
dataset = load_dataset("jfleg")
train_data = dataset["validation"]
test_data = dataset["test"]

# ==== Load Tokenizer & Model ====
print(f"Loading model and tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# ==== Format Prompts ====
def format_example(example):
    prompt = f"Correct this sentence: {example['sentence']}\nCorrected:"
    corrected = example['corrections'][0] if example['corrections'] else example['sentence']
    full_text = prompt + " " + corrected
    return tokenizer(full_text, truncation=True, padding="max_length", max_length=max_length)

print("Tokenizing datasets")
train_dataset = train_data.map(format_example, remove_columns=train_data.column_names)
test_dataset = test_data.map(format_example, remove_columns=test_data.column_names)

# ==== Training Arguments ====
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
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

print("Starting training...")
trainer.train()

# ==== Evaluation ====
print("Evaluating model...")
#text_gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# device = 0 if torch.cuda.is_available() else -1
# text_gen = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=device)

# def correct_sentence(text):
#     prompt = f"Correct this sentence: {text}\nCorrected:"
#     result = text_gen(prompt, max_new_tokens=64, do_sample=False)[0]["generated_text"]
#     return result.split("Corrected:")[-1].strip()


# Define batch generation function
def generate_batch(prompts, max_new_tokens=64):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

# Generate predictions in batches
batch_size = 16
preds = []
refs = []

for i in tqdm(range(0, len(test_data), batch_size)):
    #batch = test_data[i:i+batch_size]
    batch = test_data.select(range(i, min(i + batch_size, len(test_data))))
    print(type(batch[0]))
    prompts = [f"Correct this sentence: {ex['sentence']}\nCorrected:" for ex in batch]
    batch_outputs = generate_batch(prompts)

    for out, ex in zip(batch_outputs, batch):
        # Extract only the corrected sentence
        pred = out.split("Corrected:")[-1].strip()
        # preds.append(pred.split())  # Tokenized for BLEU
        # refs.append([ref.split() for ref in ex["corrections"]])  # Tokenized list of references
        preds.append(pred)
        refs.append(ex["corrections"])

# Evaluate with BLEU
bleu = evaluate.load("bleu")
bleu_score = bleu.compute(predictions=preds, references=refs)

print(f"\n BLEU score: {bleu_score['bleu']:.4f}")