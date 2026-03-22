# SimSCLSD: Simple Supervised Contrastive Learning for Sarcasm Detection

SimSCLSD is a two-stage training framework designed to enhance sarcasm detection in conversational contexts. By leveraging **Supervised Contrastive Learning (SCL)**, the model learns to group similar sarcastic or non-sarcastic expressions in the embedding space before fine-tuning for final classification.

## 🚀 Overview

Sarcasm is inherently nuanced and context-dependent. Standard fine-tuning with Cross-Entropy (CE) loss often struggles to capture the subtle boundary between sarcastic and literal intent. SimSCLSD addresses this by introducing a supervised contrastive pre-training phase.

### Key Features
- **Two-Stage Training**: 
    1. **Contrastive Pre-training**: Optimizes a RoBERTa-based encoder using a Supervised Contrastive Loss to pull same-label embeddings together and push different-label embeddings apart.
    2. **Classification Fine-tuning**: Fine-tunes the model with standard Cross-Entropy loss for precise boundary determination.
- **Context-Aware**: Processes conversational context (Reddit/Twitter threads) alongside the target response.
- **Robust Backbone**: Utilizes `roberta-base` as the underlying transformer architecture.

## 📂 Project Structure

```text
.
├── src/                # Source code for the model and training pipeline
│   ├── main.py         # Entry point for training and evaluation
│   ├── model.py        # SarcasmModel (RoBERTa + MLP Classifier)
│   ├── loss.py         # Supervised Contrastive Loss implementation
│   ├── dataset.py      # Data loading and preprocessing for FigLang 2020
│   ├── train.py        # Two-stage training logic
│   ├── evaluate.py     # Evaluation metrics (Precision, Recall, Macro F1)
│   └── config.json     # Hyperparameter configuration
├── thesis/             # Undergraduate thesis (LaTeX source)
│   ├── main.tex        # Thesis main document
│   └── graph_*.tex     # TikZ/PGFPlots for result visualization
├── README.md           # Project documentation
└── dataset.zip         # Raw dataset (Reddit & Twitter)
```

## 🛠️ Usage

### Installation
Ensure you have Python 3.8+ and the following dependencies installed:
```bash
pip install torch transformers scikit-learn tqdm
```

### Training the Model
Configure your hyperparameters in `src/config.json` and run the main script:
```bash
python src/main.py --config ./src/config.json
```

The script will:
1. Load the specified dataset (Reddit or Twitter).
2. Perform supervised contrastive pre-training (Stage 1).
3. Fine-tune the classifier (Stage 2).
4. Evaluate on the test set and save results in the `out/` directory.

