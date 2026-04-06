# 🎬 Multimodal Movie Genre Classification

A multimodal machine learning project that explores how **textual and visual representations** can be combined to improve **multi-label movie genre classification**.

---

## 🔍 Problem Overview

Movies often belong to **multiple genres simultaneously** (e.g., *Action + Comedy*).  
This project models genre prediction as a **multi-label classification problem**, where each genre is treated as an independent binary output.

We aim to answer:

> **How do feature representations and fusion strategies affect multimodal classification performance in a multi-label setting?**

---

## 🧠 Research Direction

We explore three core dimensions:

### 1. Feature Representation
- **Handcrafted Features**
  - Text: TF-IDF
  - Image: HOG or color histograms
- **Learned Representations**
  - Text: BERT embeddings
  - Image: CNN (ResNet)

### 2. Modality Comparison
- Text-only models
- Image-only models
- Multimodal models

### 3. Fusion Strategies
- **Late Fusion**: Combine outputs from independent models
- **Intermediate Fusion**: Combine embeddings before classification

---

## 📊 Evaluation Metrics

Since this is a **multi-label problem**, we use:

- **F1-score (sample-averaged)** → primary metric  
- **Hamming Loss** → per-label error  
- **Macro ROC-AUC** → class-wise performance  
- **Subset Accuracy** → strict match (secondary)

---

## 🤝 Team Responsibilities

### Ronald
- Text preprocessing pipeline
- TF-IDF baseline model
- BERT / transformer-based models
- Text feature extraction

### Wole
- Image preprocessing pipeline
- HOG / color histogram baseline
- CNN / ResNet models
- Image feature extraction

### Together
- Fusion models (late + intermediate)
- Evaluation and metrics implementation
- Ablation experiments
- Report + presentation

---

## 🗂️ Project Structure
movie-genre-multimodal/
│
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
│
├── data/
│ ├── raw/ # (empty in repo) → actual dataset lives in Google Drive
│ ├── interim/ # cleaned / partially processed data
│ ├── processed/ # final model-ready datasets
│ └── splits/ # train/val/test splits
│
├── notebooks/
│ ├── 01_data_exploration.ipynb
│ ├── 02_text_baseline.ipynb
│ ├── 03_image_baseline.ipynb
│ └── 04_fusion_experiments.ipynb
│
├── src/
│ ├── data/
│ │ ├── load_data.py
│ │ ├── preprocess_text.py
│ │ ├── preprocess_image.py
│ │ └── make_splits.py
│ │
│ ├── features/
│ │ ├── tfidf_features.py
│ │ ├── hog_features.py
│ │ ├── bert_embeddings.py
│ │ └── cnn_features.py
│ │
│ ├── models/
│ │ ├── text_models.py
│ │ ├── image_models.py
│ │ ├── fusion_late.py
│ │ ├── fusion_intermediate.py
│ │ └── train.py
│ │
│ ├── evaluation/
│ │ ├── metrics.py
│ │ ├── evaluate.py
│ │ └── ablation.py
│ │
│ └── utils.py
│
├── experiments/
│ ├── baseline_text.yaml
│ ├── baseline_image.yaml
│ ├── bert_text.yaml
│ ├── resnet_image.yaml
│ ├── fusion_late.yaml
│ └── fusion_intermediate.yaml
│
├── outputs/
│ ├── figures/
│ ├── logs/
│ ├── metrics/
│ └── models/
│
└── docs/
├── proposal.md
├── report_outline.md
└── presentation_outline.md


---

## 📁 Dataset Setup (IMPORTANT)

The dataset is stored in **Google Drive**, not in this repo.

Expected structure in Drive:
Movie_Genre_Project/
├── dataset_raw/
├── dataset_processed/
├── checkpoints/
├── figures/


Each team member should:
1. Download or mount the dataset locally
2. Set their local path via environment variables or config

---

## ⚙️ Environment Setup

### Option 1: pip

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Option 2: conda
```bash
conda env create -f environment.yml
conda activate movie-genre-mm
```

## Git Workflow
### Branches
- main → stable code only
- dev → integration branch

### Feature branches
- feature/text-baseline
- feature/image-baseline
- feature/bert-model
- feature/resnet-model
- feature/fusion-late
- feature/fusion-intermediate

### Workflow
1. Pull latest dev
2. Create feature branch
3. Commit work
4. Push branch
5. Open PR into dev

## Week-by-Week Plan
### Week 1
- Data exploration
- Preprocessing pipelines
- TF-IDF + HOG baselines

### Week 2
- BERT (text)
- CNN / ResNet (image)

### Week 3
- Fusion models (late + intermediate)
- Evaluation

### Week 4
- Ablation study
- Analysis
- Report + slides

---

## Key Rules
- Do not commit raw dataset to GitHub
- Keep reusable logic in src/, not notebooks
- Use consistent train/val/test splits across modalities
- Name experiments clearly
- Log results (don’t rely on memory)