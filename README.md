# 🛡️ Explainable Content Moderation System

An AI-powered content moderation system that provides transparent, interpretable decisions using transformer models and Integrated Gradients explainability.

---

## 📋 Project Overview

**Problem Statement:**
Online platforms struggle with two major challenges:
1. **Over- and under-moderation** - Inconsistent flagging damages user trust
2. **Lack of transparency** - Black-box systems provide no justification for decisions

**Solution:**
This project develops an interpretable AI system that classifies content as safe/unsafe and explains the reasoning behind each decision through an interactive dashboard.

---

## 🎯 Key Features

### 1. **High-Performance Classifier**
- **Model:** RoBERTa-base fine-tuned on 127k toxic comments
- **Performance:** 0.677 macro F1 (>20% improvement over baseline)
- **Categories:** Toxic, Severe Toxic, Obscene, Threat, Insult, Identity Hate
- **Training:** 3-fold cross-validation with per-label threshold optimization

### 2. **Explainability Module**
- **Method:** Integrated Gradients for token-level attribution
- **Output:** Human-readable explanations with semantic categorization
- **Coverage:** 100% of flagged content has explanations
- **Format:** JSON-ready for integration

### 3. **Interactive Dashboard**
- **Live Moderation:** Real-time analysis with explanations
- **Example Gallery:** Browse 100 pre-computed explanations
- **Batch Analysis:** Upload CSV files for bulk processing
- **Professional Display:** Censored profanity, clean interface

### 4. **Bias Testing**
- **Identity Bias:** 4% false positive rate
- **False Positive Rate:** 0.18-1.71% across all categories
- **Context Sensitivity:** 80% accuracy on nuanced cases
- **Comprehensive Report:** Demographic and contextual analysis

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Macro F1** | 0.677 |
| **Toxic F1** | 0.829 |
| **Obscene F1** | 0.825 |
| **Insult F1** | 0.773 |
| **Severe Toxic F1** | 0.539 |
| **Identity Hate F1** | 0.567 |
| **Threat F1** | 0.529 |
| **Identity Bias FP Rate** | 4.0% |
| **Context Sensitivity** | 80.0% |

---

## 🏗️ Project Structure
```
CMD Project 1/
├── notebooks/
│   ├── 01_data_preparation.ipynb                 # Data preparation and visualization
│   ├── 02_baseline_model.ipynb                   # Baseline model
│   ├── 03_failed_ml_models.ipynb                 # Failed Basic ML models
│   ├── 04_failed_model_bert.ipynb                # Failed BERT base model
│   ├── 05_failed_model_distilbert.ipynb          # Failed DistilBERT model
│   ├── 06_roberta_final_model.ipynb              # RoBERTa Final model
│   ├── 07_explainability.ipynb        # Integrated Gradients implementation
│   ├── 09_bias_testing.ipynb          # Fairness evaluation
│   └── artifacts/                     # Saved models & results
│       ├── primary_v3_fold0.pth
│       ├── primary_v3_fold1.pth
│       ├── primary_v3_fold2.pth
│       ├── explanation_dataset_100.json
│       ├── bias_testing_report.md
│       └── test_predictions_final.npy
├── src/
│   ├── 08_dashboard.py                    # Streamlit interactive demo
├── data/
│   └── processed_text.csv             # Training data
├── README.md                          # This file
└── requirements.txt                   # Dependencies
```

---

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/YasnaRezvani/explainable-content-moderation.git
cd explainable-content-moderation

# Install dependencies
pip install -r requirements.txt
```

### Run Dashboard
```bash
streamlit run 08_dashboard.py
```

Access at `http://localhost:8501`

### Test a Single Comment
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base", num_labels=6
)
model.load_state_dict(torch.load("notebooks/artifacts/primary_v3_fold0.pth"))
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# Predict
text = "Your text here"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
probs = torch.sigmoid(outputs.logits)[0]

labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
for label, prob in zip(labels, probs):
    print(f"{label}: {prob:.3f}")
```

---

## 🔬 Technical Details

### Model Architecture
- **Base:** RoBERTa-base (125M parameters)
- **Fine-tuning:** 2-stage (head-only → top-4 layers)
- **Loss:** Focal loss (α=0.25, γ=2.0) for class imbalance
- **Optimization:** AdamW with linear warmup
- **Hardware:** RTX 3050 Laptop (4GB VRAM)

### Explainability Method
- **Algorithm:** Integrated Gradients (Sundararajan et al., 2017)
- **Integration Steps:** 30 (balance of speed/accuracy)
- **Baseline:** Zero embeddings
- **Output:** Per-token attribution scores aggregated to full words

### Threshold Optimization
- **Method:** Per-label threshold search
- **Metric:** F1 score maximization
- **Constraint:** Minimum precision 3%
- **Search Space:** 100 thresholds per label (0.03-0.95)

---

## 🧾 Model Card (Summary)

**Model Name:** Explainable Content Moderation System (RoBERTa-based)

**Intended Use:**  
Educational and demonstrative content moderation support. Designed to assist human moderators by flagging potentially unsafe content and providing interpretable explanations. Not intended as a sole decision-maker in production moderation systems.

**Model Type:**  
Multi-label text classification with transformer architecture (RoBERTa-base).

**Training Data:**  
Jigsaw Toxic Comment Classification dataset (~127k comments). Data contains user-generated online comments labeled across six toxicity-related categories.

**Evaluation Metrics:**  
Primary metric: Macro F1 score (0.677 on held-out test set).  
Per-label F1, precision, and recall reported in the Performance Metrics section.

**Explainability:**  
Token-level attributions generated using Integrated Gradients. All flagged examples include human-readable explanations surfaced through an interactive dashboard.

**Bias & Fairness Considerations:**  
Bias testing conducted across identity-related terms and contextual variants. Observed identity false positive rate of ~4%. Known sensitivity to reclaimed terms and slang is documented.

**Limitations:**  
Reduced performance on rare classes (e.g., threat, severe toxicity). Context-dependent language and sarcasm may be misclassified. Model may flag passionate but non-toxic speech.

**Ethical Considerations:**  
Predictions should be interpreted with caution and reviewed by humans. Thresholds are optimized for balanced performance, not harm minimization in high-risk settings.

---

## 📈 Results & Insights

### What Works Well
✅ High precision on major categories (toxic, obscene, insult)  
✅ Low false positive rates (<2% for most categories)  
✅ Minimal demographic bias (4% overall)  
✅ Clear, interpretable explanations  

### Known Limitations
⚠️ Context sensitivity on slang/reclaimed terms (20% FP on "gay", "lesbian")  
⚠️ Lower performance on rare categories (threat: 0.529 F1)  
⚠️ May flag passionate but non-toxic speech  

### Future Improvements
- Fine-tune on context-aware examples (AAVE, slang, reclaimed terms)
- Ensemble with DeBERTa-v3 for performance boost
- Implement user feedback loop for continuous learning
- Add multi-language support

---

## 🎓 Educational Use

This project demonstrates:
- **Production ML Pipeline:** Data → Training → Evaluation → Deployment
- **Explainable AI:** Moving beyond black-box models
- **Bias Testing:** Proactive fairness evaluation
- **Dashboard Development:** User-friendly ML interfaces
- **Model Optimization:** Threshold tuning, cross-validation, focal loss

---

## 📚 References

### Dataset
- Jigsaw Toxic Comment Classification Challenge
- Source: `thesofakillers/jigsaw-toxic-comment-classification-challenge`

### Key Papers
- Sundararajan et al. (2017) - Integrated Gradients
- Liu et al. (2019) - RoBERTa
- Lin et al. (2017) - Focal Loss

### Tools & Libraries
- Transformers (Hugging Face)
- Captum (Explainability)
- Streamlit (Dashboard)
- PyTorch (Deep Learning)

---

## 👤 Author

**Yasna Rezvani**  
[LinkedIn](https://www.linkedin.com/in/yasna-rezvani/) | [GitHub](https://github.com/YasnaRezvani)

---

## 📄 License

This project is for educational and portfolio purposes.

---

## 🙏 Acknowledgments

- Anthropic's Claude for project guidance
- Jigsaw/Conversation AI for the dataset
- Hugging Face for transformer tools
