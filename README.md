# Devanagari Handwritten Character Recognition (OCR)

A deep learning pipeline for recognizing handwritten Devanagari characters using a custom Convolutional Neural Network (CNN), with a strong focus on evaluating model **robustness under real-world distortions** such as blur, noise, rotation, and brightness changes.

This project was developed as part of a research paper: *"Handwritten Hindi (Devanagari) Character Recognition Using Deep Learning"*.

## Overview

Most existing Devanagari OCR research reports impressive accuracy on clean, well-structured datasets — but rarely tests what happens when input images are blurry, noisy, rotated, or poorly lit, which is the norm for real-world scanned/photographed documents.

This project trains a custom CNN from scratch on the [Devanagari Handwritten Character Dataset](https://www.kaggle.com/datasets/medahmedkrichen/devanagari-handwritten-character-datase) (78,200 images, 46 classes) and systematically stress-tests it under 9 types of distortions to answer two research questions:

- **RQ1** — How well can the CNN recognize Devanagari characters on clean, well-formed input?
- **RQ2** — What happens to accuracy when real-world distortions (blur, noise, rotation, brightness) are introduced?

## Key Findings

| Distortion | Accuracy | Drop from Clean |
|---|---|---|
| Clean | **99.57%** | — |
| Blur (3x3 / 5x5) | 95.4% – 97.8% | ~1–3.5% (negligible) |
| Brightness (0.5x / 1.5x) | ~97.1% – 97.6% | <2% (negligible) |
| Rotation (±10° / ±20°) | 40% – 66% | 32% – 59% |
| Salt & Pepper Noise (5% / 10%) | 7–13% | **86–91% (catastrophic)** |

**Headline result:** A model that looks near-perfect on clean data (99.57%) can drop to near-random performance under just 5–10% salt-and-pepper noise (vs. ~2.2% random chance across 46 classes) — exposing a major real-world reliability gap that clean-accuracy benchmarks completely hide.

## Model Architecture

Custom CNN built from scratch (no pre-trained backbones like VGG/ResNet):

| Block | Details |
|---|---|
| Conv Block 1 | 2x Conv2D(32) + BatchNorm + MaxPool(2x2) + Dropout(0.25) |
| Conv Block 2 | 2x Conv2D(64) + BatchNorm + MaxPool(2x2) + Dropout(0.25) |
| Conv Block 3 | 2x Conv2D(128) + BatchNorm + MaxPool(2x2) + Dropout(0.25) |
| Dense Layers | Flatten → Dense(256) + BatchNorm + Dropout(0.5) → Dense(128) + Dropout(0.3) → Dense(46, Softmax) |

**Training Configuration:**

| Parameter | Value |
|---|---|
| Input Size | 32 x 32 x 1 (Grayscale) |
| Optimizer | Adam (initial LR = 0.001) |
| LR Schedule | ReduceLROnPlateau (factor=0.5, patience=3) |
| Loss Function | Categorical Cross-Entropy |
| Batch Size | 64 |
| Epochs | 25 (with EarlyStopping, patience=5) |
| Data Augmentation | Rotation ±10°, Shift 10%, Zoom 10% |
| Train-Test Split | 80:20 (Stratified) — 62,560 train / 15,640 test |

## Robustness Evaluation (Full Results)

The trained model was evaluated on the clean test set and on 9 artificially distorted versions of it:

| Distortion Type | Accuracy (%) | Drop (%) | Precision (%) | Recall (%) | F1-Score (%) |
|---|---|---|---|---|---|
| Clean | 98.91 | 0.00 | 98.93 | 98.91 | 98.91 |
| Blur 3x3 | 97.83 | 1.08 | 97.86 | 97.83 | 97.82 |
| Blur 5x5 | 95.36 | 3.55 | 95.66 | 95.36 | 95.37 |
| Salt & Pepper Noise 5% | 12.79 | 86.12 | 14.87 | 12.79 | 6.79 |
| Salt & Pepper Noise 10% | 7.46 | 91.45 | 1.73 | 7.46 | 1.95 |
| Rotation +10° | 66.49 | 32.42 | 77.86 | 66.49 | 64.18 |
| Rotation +20° | 42.54 | 56.37 | 58.51 | 42.54 | 39.99 |
| Rotation -10° | 40.11 | 58.80 | 61.47 | 40.11 | 36.99 |
| Brightness 1.5x | 97.64 | 1.27 | 97.78 | 97.64 | 97.66 |
| Brightness 0.5x | 97.14 | 1.77 | 97.22 | 97.14 | 97.13 |

> Note: Overall clean-data macro precision/recall/F1 = 0.9957, with the model reaching 99.57% accuracy on the held-out clean test set.

### Comparison with Related Work (Clean Accuracy)

| Method | Dataset Size | Clean Accuracy (%) |
|---|---|---|
| Deore and Pravin, VGG16 fine-tuned | 5,800 images | 96.55 |
| Sharma et al., VGG16 transfer learning | 92,000 images | 96.58 |
| Acharya et al., Deep CNN | 92,000 images | ~92.00 |
| Moudgil et al., CapsNet | Manuscripts | 94.60 |
| **Our Model, Custom CNN** | **78,200 images** | **99.57** |

> While our model outperforms prior work on clean accuracy, the central finding of this project is that **clean accuracy alone is a poor indicator of real-world OCR performance** — none of the compared methods were tested under distortion.

## Generated Plots

Running the notebook produces the following plots (saved to `results/`):

- `fig1_training.png` — Training/validation accuracy and loss curves
- `fig2_distortions.png` — Accuracy under different distortions (bar chart)
- `fig3_confusion.png` — Confusion matrix (first 15 classes)
- `fig4_top15.png` — Top 15 best-performing classes (Precision/Recall/F1)
- `fig5_bottom15.png` — Bottom 15 most-confused classes
- `fig6_samples.png` — Sample character under each distortion type

## Dataset

[Devanagari Handwritten Character Dataset](https://www.kaggle.com/datasets/medahmedkrichen/devanagari-handwritten-character-datase) — 32x32 grayscale images, 78,200 samples across 46 character classes (36 consonants + 10 digits).

## How to Run

This notebook is designed for **Google Colab** (uses Kaggle API for dataset download, GPU recommended — trained on T4 in ~12 minutes for 25 epochs).

1. Open `devanagari_character_recognition.ipynb` in Google Colab.
2. Get your `kaggle.json` API key from [Kaggle Account Settings](https://www.kaggle.com/settings) → API → "Create New Token".
3. Run the first cell — it will prompt you to upload `kaggle.json` and automatically download/extract the dataset.
4. Run all remaining cells in order to train the model and generate evaluation plots/metrics.

### To run locally instead:
- Skip the Kaggle/Colab upload cell.
- Manually download the dataset and set `DATA_DIR` to its local path.
- Install dependencies from `requirements.txt`.

## Requirements

```bash
pip install -r requirements.txt
```

Key libraries: TensorFlow, OpenCV, scikit-learn, matplotlib, seaborn.

## Project Structure

```
devanagari-ocr/
├── README.md
├── requirements.txt
├── devanagari_character_recognition.ipynb
└── results/              # Generated plots (after running notebook)
```

## Limitations & Future Work

- Evaluated on isolated characters only — real-world OCR needs word/sentence-level recognition.
- Distortions are synthetically applied; real-world degradation can be more unpredictable and compounded.
- Rotation robustness could improve with wider augmentation ranges or rotation-invariant architectures (e.g., Capsule Networks).
- Noise robustness is the most critical gap — training with noise-augmented data is a clear next step.

## Author

Sanchi — B.Tech (AI/ML), Final Year
Department of Information Technology, Indira Gandhi Delhi Technical University for Women
