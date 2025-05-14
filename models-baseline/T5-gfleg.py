import torch, re, numpy as np, pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and filter JFLEG (sentences < 20 tokens w/ punctuation)
jfleg = load_dataset("jfleg", split="test")
def count_tokens(text): return len(re.findall(r'\w+|[^\w\s]', text))
jfleg_short = jfleg.filter(lambda ex: count_tokens(ex["sentence"]) < 20)
jfleg_short = jfleg_short.select(range(min(10000, len(jfleg_short))))
inputs = [ex["sentence"] for ex in jfleg_short]
refs = [[ex["corrections"]] for ex in jfleg_short]

gleu_metric = evaluate.load("google_bleu")
t5_models = ["t5-small", "t5-base"]
t5_results = []

for model_name in t5_models:
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()

    preds = []
    for text in tqdm(inputs, desc=f"Evaluating {model_name}"):
        input_text = "correct: " + text
        input_ids = tokenizer(input_text, return_tensors="pt", truncation=True).input_ids.to(device)
        with torch.no_grad():
            outputs = model.generate(input_ids, max_length=128)
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        preds.append(pred)

    gleu = gleu_metric.compute(predictions=preds, references=refs)
    acc = np.mean([pred.strip() in map(str.strip, ref[0]) for pred, ref in zip(preds, refs)])
    t5_results.append({
        "Model": model_name,
        "GLEU": round(gleu["google_bleu"], 4),
        "Accuracy": round(acc, 4)
    })

print("\nT5 Results:")
print(pd.DataFrame(t5_results).to_markdown(index=False))
