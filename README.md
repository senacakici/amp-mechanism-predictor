# AMP Mechanism Predictor

**Multi-label classification of antimicrobial peptide (AMP) mechanisms of action**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Motivation

Most AMP prediction tools ask: *is this peptide antimicrobial?*  
This project asks: *how does it work?*

Antimicrobial peptides kill bacteria through diverse and often simultaneous mechanisms membrane disruption, protein synthesis inhibition, cell division interference, metabolic disruption, and more. Standard single-label classifiers collapse this complexity into misleading categories. This project builds a **multi-label** framework that predicts the full mechanistic profile of an AMP from its sequence.

A key challenge: the training data is historically mislabelled (most AMPs were incorrectly assigned to "membrane disruption" for decades). This codebase includes tools to handle **noisy, partial, and conflicting labels** explicitly.

---

## Mechanism Labels

The model predicts across 8 mechanistic classes:

| Label | Description |
|-------|-------------|
| `membrane_disruption` | Pore formation, membrane lysis, lipid phase disruption |
| `membrane_depolarization` | Loss of membrane potential without structural disruption |
| `cell_wall_synthesis` | Inhibition of peptidoglycan synthesis |
| `protein_synthesis` | Ribosome targeting or translation inhibition |
| `dna_rna_targeting` | Nucleic acid binding or replication inhibition |
| `cell_division` | FtsZ or divisome interference |
| `metabolic_disruption` | Enzyme inhibition, ATP depletion |
| `immunomodulatory` | Host immune system modulation |

---

## Pipeline Overview

```
Peptide sequence
      │
      ▼
Feature extraction
  ├── Physicochemical (charge, hydrophobicity, amphipathicity)
  ├── AAC / DPC descriptors
  └── ESM-2 protein language model embeddings (optional)
      │
      ▼
Multi-label classifier
  ├── Binary Relevance baseline
  ├── Classifier Chain
  └── Label-aware MLP with BCE + label co-occurrence loss
      │
      ▼
Mechanism profile + confidence scores
```

---

## Quickstart

```bash
git clone https://github.com/[yourusername]/amp-mechanism-predictor
cd amp-mechanism-predictor
pip install -r requirements.txt

# Download and prepare public AMP data
python src/data/prepare_dataset.py

# Train baseline model
python src/models/train.py --model binary_relevance --features physicochemical

# Evaluate
python src/evaluation/benchmark.py --model_path results/model.pkl
```

---

## Data Sources

Public datasets used in this project:

- **APD3** — Antimicrobial Peptide Database (Wang et al., 2016)
- **DBAASP** — Database of Antimicrobial Activity and Structure of Peptides
- **CAMPR4** — Collection of Anti-Microbial Peptides

Mechanistic annotations are derived from literature curation with ontology-informed label assignment. See `data/README.md` for details.

---

## Project Structure

```
amp-mechanism-predictor/
├── data/                    # Raw and processed datasets
│   └── README.md
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   └── 03_model_comparison.ipynb
├── src/
│   ├── data/
│   │   ├── prepare_dataset.py   # Download + preprocess public AMPs
│   │   ├── features.py          # Feature extraction
│   │   └── label_curation.py    # Ontology-informed label assignment
│   ├── models/
│   │   ├── train.py             # Training script
│   │   ├── baseline.py          # Binary relevance, classifier chain
│   │   └── mlp_multilabel.py    # Label-aware MLP
│   └── evaluation/
│       ├── benchmark.py         # Full evaluation pipeline
│       └── metrics.py           # Multi-label metrics
├── tests/
├── results/
├── requirements.txt
└── README.md
```

---

## Evaluation Metrics

Multi-label classification requires metrics beyond accuracy:

- **Hamming Loss** — fraction of incorrectly predicted labels
- **Subset Accuracy** — exact match across all labels
- **Macro/Micro F1** — per-label and aggregate F1
- **Label Ranking Average Precision (LRAP)**
- **Coverage Error** — how far down the ranked list to include all true labels

---

## Background Reading

- Wenzel & Schäfer (2020). *A How-To Guide for Mode of Action Analysis of Antimicrobial Peptides.* Frontiers in Cellular and Infection Microbiology.
- Müller et al. (2016). *Daptomycin inhibits cell envelope synthesis by interfering with fluid membrane microdomains.* PNAS.
- Wenzel et al. (2018). *The multifaceted antibacterial mechanisms of tyrocidine and gramicidin S.* mBio.

---

## Roadmap

- [x] Feature extraction pipeline (physicochemical + AAC)
- [x] Baseline multi-label classifiers
- [x] Evaluation framework with multi-label metrics
- [ ] ESM-2 embedding integration
- [ ] Label noise modelling (partial label learning)
- [ ] Ontology-based label hierarchy
- [ ] Web demo (Gradio/Streamlit)


