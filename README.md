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

---

## 📂 Project Structure

```text
├── models/
│   ├── utils/              
│   ├── builder.py          
│   ├── causal_mil_loss.py  
│   ├── ctran.py            
│   ├── modules.py               
│   ├── MIL_models.py       
│   └── resnet_custom.py    
├── utils/
│   └── utils.py            
├── wsi_core/
│   ├── dataset_modules/    
│   ├── wsi_core/           
│   ├── create_patches_fp.py 
│   └── extract_features_fp.py 
└── README.md
