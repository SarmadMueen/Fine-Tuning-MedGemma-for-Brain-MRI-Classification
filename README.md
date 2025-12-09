# Fine-Tuning MedGemma for Brain MRI Diagnosis

This project fine-tunes Google's **MedGemma-4b** Vision-Language Model (VLM) to classify Brain MRI scans into specific cancer categories using Low-Rank Adaptation (LoRA) and Supervised Fine-Tuning (SFT).

## Project Overview

This repository adapts the `medgemma-4b-it` model to analyze MRI scans and identify brain tumor types. By fine-tuning the model on domain-specific medical imagery, classification accuracy was improved significantly over the baseline zero-shot performance.

**Key Technical Features**

* **Multimodal Architecture:** Processes visual data and text prompts simultaneously.
* **Efficient Training:** Utilizes LoRA to fine-tune on consumer hardware with reduced VRAM usage.
* **SFT Pipeline:** Implements the Hugging Face `TRL` library for streamlined training.

## Dataset

The model was trained on the **Brain Cancer MRI Dataset** sourced from Kaggle.

* **Source:** [Kaggle: Brain Cancer MRI Dataset](https://www.kaggle.com/datasets/orvile/brain-cancer-mri-dataset)
* **Classes:**
  1. `brain_glioma`
  2. `brain_menin` (Meningioma)
  3. `brain_tumor` (Pituitary/General)

## Technical Implementation

1. **Preprocessing:** Images are converted to RGB and resized. Prompts are formatted as a Visual Question Answering (VQA) task.
2. **Model Configuration:**
   * **Base Model:** `google/medgemma-4b-it`
   * **Quantization:** Loaded in `bfloat16` (supports 4-bit QLoRA).
   * **PEFT:** LoRA rank `r=16`, `alpha=16`, targeting linear layers.
3. **Training:** Optimized using `adamw_torch_fused` with a batch size of 8 and gradient accumulation.

## Results

Performance was measured on a held-out validation set after one epoch.

| Metric | Base Model (Zero-Shot) | Fine-Tuned Model |
| :--- | :---: | :---: |
| **Accuracy** | 33.7% | **89.3%** |
| **F1 Score** | 0.17 | **0.89** |

## Installation and Usage

### Prerequisites
* Python 3.10+
* GPU with 16GB+ VRAM (e.g., T4, A10)
* Hugging Face and Kaggle Accounts

### Setup

1. **Clone the repository**
   ```bash
   git clone [https://github.com/your-username/medgemma-brain-mri.git](https://github.com/your-username/medgemma-brain-mri.git)
   cd medgemma-brain-mri
