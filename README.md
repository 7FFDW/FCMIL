# FC-MIL: Frequency-Aware Causal Regularization for MIL

**Official PyTorch Implementation of the Paper:**
> **Frequency-Aware Causal Regularization for Multiple Instance Learning in Whole Slide Image Classification**

---

![main](https://github.com/user-attachments/assets/8c0d5753-1fb1-4248-bdcd-ba70eb10999d)


## 🚀 Overview
FC-MIL is a novel Multiple Instance Learning (MIL) framework designed to address two fundamental limitations in gigapixel Whole Slide Image (WSI) analysis:
1. **Micro-texture Neglect**: Traditional MIL often overlooks fine-grained pathological textures.
2. **Spurious Causal Correlations**: Confounding biases (e.g., stroma, artifacts) often lead to false-positive predictions.

Our framework integrates **Frequency-aware Attention (FAA)** to capture diagnostic textures and **Causal Regularization (CR)** to eliminate non-diagnostic biases via counterfactual intervention.

---

## ✨ Key Contributions

### 1. Frequency-Aware Attention (FAA)
* **Frequency Decoupling**: Uses 1D-FFT to separate instance sequences into structural (low-freq) and textural (high-freq) components.
* **Dynamic Activation Block (DAB)**: Adaptively amplifies diagnostic signals in high-frequency bands to identify subtle lesions.

### 2. Causal Regularization (CR)
* **Backdoor Adjustment**: Implements a $do(X)$ intervention strategy.
* **Counterfactual Tasks**: Uses **Drop-TopK** (removing key evidence) and **Re-BottomK** (replacing irrelevant background) to ensure the model focuses on true causal pathological regions.

### 3. Efficiency & Scalability
* Optimized for large-scale WSIs (up to 50k+ instances per slide).
* Supports **Gradient Checkpointing** and **Flash Attention** for low memory footprint during training and inference.

---

## 📂 Project Structure

```text
├── models/
│   ├── utils/              # Model-specific internal utilities
│   ├── builder.py          # Model construction factory
│   ├── causal_mil_loss.py  # Causal Regularization loss logic
│   ├── ctran.py            # C-Tran backbone implementation
│   ├── FC_MIL.py           # Core architecture (FCMIL, FAA, DABlock)
│   ├── MaxPooling.py       # Max pooling baseline
│   ├── MeanPooling.py      # Mean pooling baseline
│   ├── MIL_models.py       # Standard MIL baselines (ABMIL, CLAM, etc.)
│   └── resnet_custom.py    # Custom ResNet feature extractor
├── utils/
│   └── utils.py            # General helper functions and metrics
├── wsi_core/
│   ├── dataset_modules/    # Data loading and augmentation logic
│   ├── wsi_core/           # WSI processing core modules
│   ├── create_patches_fp.py # Patch tiling and preprocessing
│   └── extract_features_fp.py # Feature extraction pipeline
└── README.md
