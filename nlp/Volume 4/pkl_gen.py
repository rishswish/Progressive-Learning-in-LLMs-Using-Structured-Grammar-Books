import os
import re
import pickle
import stanza
from docx import Document


# ─── CONFIG ───────────────────────────────────────────────────────────────────
input_folder  = r'D:\AI-ML\Github Projects\EY Dataset\NLU-Project\nlp\Volume 4\Lessons'
output_folder = r'D:\AI-ML\Github Projects\EY Dataset\NLU-Project\nlp\Volume 4\Output'
os.makedirs(output_folder, exist_ok=True)
# ────────────────────────────────────────────────────────────────────────────────

# ─── INIT STANZA ───────────────────────────────────────────────────────────────
stanza.download('en', verbose=False)
nlp = stanza.Pipeline(
    lang='en',
    processors='tokenize,pos,lemma,depparse,ner',
    tokenize_no_ssplit=True,
    use_gpu=True
)
# ────────────────────────────────────────────────────────────────────────────────

# ─── PROCESS DOCX FILES ────────────────────────────────────────────────────────
pattern = re.compile(r'v(\d+)[-_]l\s*\((\d+)\)\.docx$', re.IGNORECASE)

for fn in sorted(os.listdir(input_folder)):
    match = pattern.match(fn)
    if not match:
        # skip any files that don’t follow v{vol}-l ({lesson}).docx
        continue

    vol_num, lesson_num = match.groups()
    out_name = f"v{vol_num}_l{lesson_num}.pkl"
    out_path = os.path.join(output_folder, out_name)

    # load the Word doc
    doc_path = os.path.join(input_folder, fn)
    doc = Document(doc_path)

    # parse each non-blank paragraph
    parsed_paras = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            parsed_paras.append(nlp(text))

    # pickle the list of Stanza Document objects
    with open(out_path, 'wb') as fout:
        pickle.dump(parsed_paras, fout)

    print(f"Saved {out_name} ({len(parsed_paras)} paras)")
# ────────────────────────────────────────────────────────────────────────────────

print("All done!")  
