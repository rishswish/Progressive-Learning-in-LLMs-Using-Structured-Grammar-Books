import os
import sys
import json
import pickle
from tqdm.auto import tqdm
import stanza

# ─── Read volume number from command line (default = 3) ───
try:
    volume = int(sys.argv[1]) if len(sys.argv) > 1 else 3
except ValueError:
    print("Invalid volume; using default = 3.")
    volume = 3

# ─── Load your JSON for this volume ───
json_path = f"volume{volume}_lessons.json"
with open(json_path, "r", encoding="utf-8") as f:
    volumes = json.load(f)

# ─── Stanza pipeline ───
stanza.download('en', verbose=False)
nlp = stanza.Pipeline(
    lang='en',
    processors='tokenize,pos,depparse,lemma,ner',
    tokenize_no_ssplit=True,
    use_gpu=False
)

os.makedirs("./lessons", exist_ok=True)

# ─── Parse & pickle each lesson ───
for lesson_key, sentences in tqdm(volumes.items(),
                                  desc="Lessons",
                                  total=len(volumes)):
    parsed_docs = []
    for sent in tqdm(sentences,
                     desc=f"  {lesson_key}",
                     leave=False,
                     unit="sent"):
        parsed_docs.append(nlp(sent))

    out_path = f"./lessons/{lesson_key}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(parsed_docs, f)
