import torch, numpy as np, pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reuse the inputs and refs from the same filtering above
gleu_metric = evaluate.load("google_bleu")
gpt_models = ["gpt2", "EleutherAI/gpt-neo-125M", "gpt2-medium"]
gpt_results = []

for model_name in gpt_models:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    preds = []
    for text in tqdm(inputs, desc=f"Evaluating {model_name}"):
        prompt = "Correct the grammar: " + text
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.to(device)
        with torch.no_grad():
            outputs = model.generate(input_ids, max_length=input_ids.shape[1] + 32)
        generated = outputs[0][input_ids.shape[1]:]
        pred = tokenizer.decode(generated, skip_special_tokens=True)
        preds.append(pred)

    gleu = gleu_metric.compute(predictions=preds, references=refs)
    acc = np.mean([pred.strip() in map(str.strip, ref[0]) for pred, ref in zip(preds, refs)])
    gpt_results.append({
        "Model": model_name,
        "GLEU": round(gleu["google_bleu"], 4),
        "Accuracy": round(acc, 4)
    })

print("\nGPT Results:")
print(pd.DataFrame(gpt_results).to_markdown(index=False))
