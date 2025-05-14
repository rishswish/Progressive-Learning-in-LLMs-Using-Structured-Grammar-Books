import os
import json
import sys
from pathlib import Path
from docx import Document

def extract_lessons(folder_path, volume=2, n_lessons=95, output_path=None):
    """
    Extract up to `n_lessons` from files named
      Volume{volume}-Lesson1.docx … Volume{volume}-Lesson{n_lessons}.docx
    in `folder_path`, and write them to JSON.
    """
    if output_path is None:
        output_path = f"volume{volume}_lessons.json"

    lessons = {}

    for i in range(1, n_lessons + 1):
        key = f"v{volume}_l{i}"
        docx_path = Path(folder_path) / f"Volume{volume}-Lesson{i}.docx"

        if not docx_path.exists():
            print(f"Warning: {docx_path} not found, skipping.")
            continue

        doc = Document(docx_path)
        # collect non-empty, stripped paragraphs
        paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        lessons[key] = paras

    # write out JSON
    with open(output_path, "w", encoding="utf-8") as fout:
        json.dump(lessons, fout, ensure_ascii=False, indent=2)
    print(f"Extracted {len(lessons)} lessons → {output_path}")


if __name__ == "__main__":
    # usage: python this_script.py [volume_number]
    # default volume=3
    try:
        volume = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    except ValueError:
        print("Volume must be an integer; using default 3.")
        volume = 3

    # folder is fixed as "doc-output"
    extract_lessons("doc-output", volume=volume)
