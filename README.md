# 📚 Progressive Learning in LLMs Using Structured Grammar Books

This project explores a structured curriculum-based approach to pretraining large language models (LLMs) using syntactically rich grammar textbooks. By leveraging linguistic features and progressive learning strategies, we improve model generalization, syntactic competence, and sample efficiency—especially in low-resource settings.

---

## 🧠 Motivation

Traditional LLMs rely on large, unstructured corpora, which can lead to inefficiencies and poor syntactic awareness. This work proposes an alternative: a structured, human-like progression through grammar curricula to:

- Improve syntactic generalization
- Reduce data and compute requirements
- Enhance fine-tuning performance on downstream tasks

---

## 📘 Dataset

The dataset was built from the **New Concept English** textbook series (Volumes 1–4). Key steps:

- OCR-based digitization (Tesseract)
- Sentence normalization and correction
- POS, dependency, and morphological tagging (Stanza)
- 18,067 labeled sentences
- Lesson-level structuring for curriculum-based training

---

## 🛠️ Methodology

### 📐 Model Selection

| Model     | Params (M) | Hidden Size | Layers | Heads |
|-----------|-------------|-------------|--------|--------|
| GPT-2     | ~117        | 768         | 12     | 12     |
| T5-base   | ~220        | 768         | 12     | 12     |

### 🔧 Training Strategy

1. **From Scratch**: No pretrained weights used.
2. **Curriculum Pretraining**:
   - ~345 lessons
   - Each contains 20–70 sentences
   - Lessons reinforce earlier syntax with loops
   - Pedagogically structured → easier-to-learn progression

3. **Fine-tuning on TREC**:
   - Adapt models to TREC question classification
   - Use label sets like ENTY, ABBR, LOC, HUM, NUM

### 🧩 Syntax Feature Embedding

- POS tags, dependency relations, morphological features
- Token embeddings fused with [Token ∥ Syntax] structure
- Shown to improve downstream generalization

---

## 🧪 Results & Evaluation

### 🎯 TREC Classification Performance

- **SyntaxGPT**: 63% accuracy on TREC (vs. 95% for GPT-2 baseline)
- During curriculum training: F1 scores increased from 0.58 → 0.70
- Ablation studies:
  - Syntax embeddings improved generalization
  - Faster convergence vs. control

### 📈 Learning Curves

- Curriculum-based training led to **stable and monotonic** performance gains
- Syntax-aware models were more sample-efficient and domain-resilient

---

## 📊 Poster

![Project Poster](images/nlu_poster.jpg)

---

## 🚧 Limitations & Future Work

### Known Limitations
- **Curriculum Saturation**: Diminishing returns at high complexity
- **Domain Scope**: Grammar books limit real-world generalization
- **Data Scale**: Small corpora limit scale-up
- **Decoder Constraints**: GPT-2 struggled vs. encoder-decoder models

### Future Directions
- Expand syntactic and semantic diversity
- Integrate morphological enrichment (e.g., Wiktionary)
- Conduct ablation studies on curriculum structure
- Apply framework to multilingual grammar curricula (e.g., CGC Thesis)

---

## 🧾 References

1. Bengio et al. (2009). Curriculum Learning. ICML.
2. Elman, J. (1993). Learning and development in recurrent neural networks.
3. Piantadosi, S. (2023). Human inductive biases as rational inference.
4. Reali & Christiansen (2005). Richness of morphological structure.

---

## 👨‍🔬 Authors

- Airu Liu  
- Mustafa Poonawala  
- Devyani Hebbar  
- **Rishabh Patil**


