import os
import re
import json
import pytesseract
from PIL import Image
import logging
import argparse

# Predefined list of abbreviations (for protecting their dots during sentence splitting)
ABBREVIATIONS = [
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "St.",
    "a.m.", "A.M.", "p.m.", "P.M.",
    "b.c.", "B.C.",
    "i.e.", "e.g.", "etc.", "vs.",
    "Ph.D.", "Ph.d."
]

# Predefined answer choices for 60 lessons (example values)
choices = [
    ['d', 'a', 'c', 'c', 'd', 'b', 'd', 'd', 'c', 'b', 'b', 'a'],
    ['d', 'b', 'c', 'b', 'd', 'b', 'c', 'a', 'b', 'a', 'a', 'c'],
    ['d', 'd', 'a', 'd', 'b', 'c', 'b', 'b', 'a', 'd', 'b', 'b'],
    ['a', 'c', 'd', 'b', 'd', 'c', 'b', 'c', 'd', 'b', 'c', 'a'],
    ['c', 'b', 'a', 'b', 'c', 'd', 'b', 'c', 'c', 'b', 'd', 'd'],
    ['b', 'a', 'c', 'c', 'b', 'a', 'a', 'd', 'a', 'a', 'b', 'c'],
    ['b', 'd', 'b', 'a', 'c', 'b', 'c', 'a', 'a', 'd', 'b', 'b'],
    ['c', 'c', 'd', 'd', 'a', 'a', 'c', 'c', 'b', 'a', 'd', 'c'],
    ['a', 'd', 'a', 'c', 'b', 'd', 'b', 'a', 'b', 'c', 'c', 'a'],
    ['d', 'c', 'a', 'b', 'd', 'd', 'b', 'd', 'a', 'c', 'b', 'a'],
    ['c', 'c', 'a', 'd', 'c', 'b', 'a', 'a', 'c', 'a', 'd', 'b'],
    ['c', 'd', 'a', 'b', 'd', 'c', 'b', 'c', 'b', 'd', 'd', 'a'],
    # ... (add all 60 lists accordingly) ...
]

def protect_abbreviations(text):
    """Replace the dot in each abbreviation with a placeholder token."""
    for abbr in ABBREVIATIONS:
        placeholder = abbr.replace('.', '<<DOT>>')
        text = text.replace(abbr, placeholder)
    return text

def unprotect_abbreviations(text):
    """Revert the placeholder token back to a dot."""
    for abbr in ABBREVIATIONS:
        placeholder = abbr.replace('.', '<<DOT>>')
        text = text.replace(placeholder, abbr)
    return text

def ocr_image(image_path, lang="eng+chi_sim"):
    try:
        image = Image.open(image_path)
        return pytesseract.image_to_string(image, lang=lang)
    except Exception as e:
        logging.error(f"Error processing {image_path}: {e}")
        return None

def remove_spaces_between_chinese(text):
    pattern = re.compile(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])')
    while pattern.search(text):
        text = pattern.sub(r'\1\2', text)
    return text

def process_text(text):
    text = re.sub(r'(_+)[.-](_+)', '______', text)
    text = re.sub(r'_{2,}', '______', text)
    text = re.sub(r'(______)[ \t]+([,\.!\?，。！？])', r'\1\2', text)
    text = remove_spaces_between_chinese(text)
    text = re.sub(r'([A-Za-z0-9])([\u4e00-\u9fff])', r'\1 \2', text)
    text = re.sub(r'([\u4e00-\u9fff])([A-Za-z0-9])', r'\1 \2', text)
    text = re.sub(r'[ \t]*([，。！？；：])[ \t]*', r'\1', text)
    lines = text.splitlines()
    non_blank_lines = [line for line in lines if line.strip() != ""]
    return "\n".join(non_blank_lines)

def extract_article(combined_text):
    raw_start_marker = "然后回答以下问题。"
    raw_end_marker = "生词和短语"
    start_idx = combined_text.find(raw_start_marker)
    if start_idx != -1:
        start_idx += len(raw_start_marker)
        end_idx = combined_text.find(raw_end_marker, start_idx)
        if end_idx != -1:
            article_text = combined_text[start_idx:end_idx].strip()
            article_text = re.sub(r'\s+', ' ', article_text)
            # Protect abbreviations so dots inside them aren't used as sentence delimiters.
            article_text = protect_abbreviations(article_text)
            sentences = re.findall(r'[^.!?]+[.!?]', article_text)
            return [unprotect_abbreviations(s.strip()) for s in sentences if s.strip()]
    return []

def parse_multi_choice(text):
    def clean_trailing_parenthetical(txt):
        return re.sub(r'\s*\(.*?\)\s*[\.\!\?]?\s*$', '', txt)
    questions = []
    blocks = re.split(r'(?=^\d+)', text, flags=re.M)
    for block in blocks:
        block = block.strip()
        if not re.match(r'^\d+', block):
            continue
        num_match = re.match(r'^(\d+)[\.\)]?\s+', block)
        if num_match:
            q_number = num_match.group(1)
            block = block[num_match.end():]
        else:
            q_number = ""
        parts = block.split("(a)")
        if len(parts) < 2:
            continue
        question_text = parts[0].strip()
        question_text = clean_trailing_parenthetical(question_text)
        question_text = re.sub(r'^\d+[\.\)]?\s+', '', question_text)
        # Replace newlines with a space.
        question_text = question_text.replace("\n", " ")
        choices_text = "(a)" + "(a)".join(parts[1:])
        choice_a = re.search(r'\(a\)\s*(.*?)\s*(?=\(b\)|\n\d|$)', choices_text, re.S)
        choice_b = re.search(r'\(b\)\s*(.*?)\s*(?=\(c\)|\n\d|$)', choices_text, re.S)
        choice_c = re.search(r'\(c\)\s*(.*?)\s*(?=\(d\)|\n\d|$)', choices_text, re.S)
        choice_d = re.search(r'\(d\)\s*(.*?)(?=\n\d|$)', choices_text, re.S)
        a_text = clean_trailing_parenthetical(choice_a.group(1).splitlines()[0].strip()) if choice_a else ""
        b_text = clean_trailing_parenthetical(choice_b.group(1).splitlines()[0].strip()) if choice_b else ""
        c_text = clean_trailing_parenthetical(choice_c.group(1).splitlines()[0].strip()) if choice_c else ""
        d_text = clean_trailing_parenthetical(choice_d.group(1).splitlines()[0].strip()) if choice_d else ""
        question_dict = {
            "number": q_number,
            "question": question_text,
            "a": a_text,
            "b": b_text,
            "c": c_text,
            "d": d_text
        }
        questions.append(question_dict)
    return questions

def extract_multi_choice(combined_text):
    raw_marker = "多项选择题"
    marker_idx = combined_text.find(raw_marker)
    if marker_idx != -1:
        mc_text = combined_text[marker_idx + len(raw_marker):].strip()
        return parse_multi_choice(mc_text)
    return []

def create_lessons(input_dir, output_dir, lang="eng+chi_sim"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    supported_formats = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
    image_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)
                           if f.lower().endswith(supported_formats)])
    lesson_count = 0
    for i in range(0, len(image_files), 4):
        lesson_count += 1
        lesson_images = image_files[i:i+4]
        combined_text = ""
        for image_path in lesson_images:
            raw_text = ocr_image(image_path, lang=lang)
            if raw_text is None:
                raw_text = "OCR failed for this page."
            cleaned_text = process_text(raw_text)
            combined_text += cleaned_text + "\n\n"
        article_list = extract_article(combined_text)
        multi_choice_list = extract_multi_choice(combined_text)
        lesson_json = {
            "lesson": lesson_count,
            "article": article_list,
            "multi-choice": multi_choice_list,
            "choices-answer": choices[lesson_count - 1] if lesson_count <= len(choices) else []
        }
        json_filename = os.path.join(output_dir, f"lesson_{lesson_count}.json")
        with open(json_filename, "w", encoding="utf-8") as jf:
            json.dump(lesson_json, jf, ensure_ascii=False, indent=4)
        logging.info(f"Saved lesson JSON to {json_filename}")

def main():
    parser = argparse.ArgumentParser(
        description="Post-process OCR output: update multi-choice questions (replace blanks with correct answers) and create lesson JSON files."
    )
    parser.add_argument("input_dir", help="Directory containing image files.")
    parser.add_argument("output_dir", help="Output folder for lesson JSON files.")
    parser.add_argument("--lang", default="eng+chi_sim", help="Tesseract languages (default: eng+chi_sim).")
    args = parser.parse_args()
    create_lessons(args.input_dir, args.output_dir, lang=args.lang)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
