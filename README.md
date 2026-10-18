# OccuScan AI: Ocular Disease Recognition

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python)](https://www.python.org/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg?logo=tensorflow)](https://tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Primary Pipeline](https://img.shields.io/badge/Primary%20Pipeline-Resnet50.ipynb-emerald.svg)](Resnet50.ipynb)


**OccuScan AI** is an end-to-end ophthalmic computer vision and clinical decision support system designed to screen, classify, and localize retinal abnormalities from digital fundus photographs. 

Powered by a fine-tuned **ResNet50 Deep Residual Convolutional Network** trained on the ODIR (Ocular Disease Intelligent Recognition) dataset, the platform identifies **8 primary ocular condition categories** with high Area Under the ROC Curve (AUC) performance, provides **Grad-CAM interpretability heatmaps**, and delivers an interactive clinical web interface.

---

## 🌟 Key Features

* **Primary ResNet50 Deep Learning Pipeline (`Resnet50.ipynb`)**:
  * Transfer learning from ImageNet pre-trained ResNet-50 backbone.
  * Spatial feature pooling via `GlobalAveragePooling2D()` to eliminate dense parameter explosion.
  * Regularized multi-tier classification head with Batch Normalization and Dropout ($0.3$).
  * **Square-Root Smoothed Class Balancing**: Dynamically penalizes majority classes while dampening extreme gradient shocks from rare diseases.
  * Checkpoint optimization tracking `val_auc` with early stopping and automatic weight restoration.
* **8-Class Retinal Pathology Classification**:
  * Normal, Diabetes (Diabetic Retinopathy), Glaucoma, Cataract, AMD (Age-related Macular Degeneration), Hypertension, Myopia, and Others.
* **Explainable AI (XAI)**:
  * Grad-CAM (Gradient-weighted Class Activation Mapping) visually overlays salient retinal regions (optic disc, macula, vascular arcades, exudates/drusen).
* **Clinical Web Platform & REST API (`app.py`)**:
  * High-performance asynchronous FastAPI server.
  * Diagnostic report dashboard with interactive risk meters, urgency tags, and follow-up guidance.
  * Batch processing endpoint with real-time JSON payload outputs.
* **Command-Line Suite**:
  * Direct CLI inference script (`predict.py`) and automated dataset validation script (`batch_test.py`).

---

## 🏗️ Model Architecture (`Resnet50.ipynb`)

`Resnet50.ipynb` serves as the primary training notebook and core model generation pipeline. It directly produces the deployed production model: `models/best_resnet50_model.keras`.

```
                    Input Fundus Photograph (224 × 224 × 3)
                                       │
                                       ▼
                   ┌────────────────────────────────────────┐
                   │   ResNet50 Feature Extractor           │
                   │   (Pretrained on ImageNet, Frozen)     │
                   └────────────────────────────────────────┘
                                       │
                                       ▼
                   ┌────────────────────────────────────────┐
                   │   GlobalAveragePooling2D()             │
                   │   (Aggregates 2048 spatial feature maps│
                   └────────────────────────────────────────┘
                                       │
                                       ▼
                   ┌────────────────────────────────────────┐
                   │   BatchNormalization()                 │
                   └────────────────────────────────────────┘
                                       │
                                       ▼
                   ┌────────────────────────────────────────┐
                   │   Dense(256, activation='relu')        │
                   │   Dropout(0.30)                        │
                   └────────────────────────────────────────┘
                                       │
                                       ▼
                   ┌────────────────────────────────────────┐
                   │   Dense(128, activation='relu')        │
                   └────────────────────────────────────────┘
                                       │
                                       ▼
                   ┌────────────────────────────────────────┐
                   │   Dense(8, activation='softmax')       │
                   └────────────────────────────────────────┘
                                       │
                                       ▼
               8-Class Categorical Probability Distribution
```

### Why This Architecture?

1. **Spatial Aggregation via GlobalAveragePooling2D**:
   Replacing legacy `Flatten()` operations with `GlobalAveragePooling2D()` reduces trainable parameters from ~75M+ to under 1M in the classification head, preventing spatial overfitting and speeding up inference by over $4\times$.
2. **Square-Root Dampened Class Weighting**:
   Medical datasets often exhibit extreme label imbalance. Rather than standard inverse frequency (which can trigger gradient explosion), `Resnet50.ipynb` utilizes square-root dampening:
   $$\text{Weight}_c = \sqrt{\frac{N}{C \times N_c}}$$
   This penalizes false negatives on minority pathologies (e.g., Cataract, AMD, Hypertension) while maintaining optimization stability.
3. **AUC-Driven Early Stopping**:
   Monitors validation AUC (`val_auc`) with `patience=4` to capture the checkpoint with optimal multi-class discriminative power.

---

## 🎯 Supported Ocular Pathologies

The model classifies fundus photographs into 8 clinically recognized categories:

| ID | Class Name | Clinical Hallmarks & Features Detected | Urgency Level |
|---|---|---|---|
| `0` | **Age-related Macular Degeneration (AMD)** | Drusen deposits, geographic atrophy, choroidal neovascularization | Specialist Referral |
| `1` | **Cataract** | Lens opacity, media haziness, loss of retinal vascular crispness | Consultation Advised |
| `2` | **Diabetes (Diabetic Retinopathy)** | Microaneurysms, cotton-wool spots, hard exudates, retinal hemorrhages | High Priority |
| `3` | **Glaucoma** | Optic cup-to-disc ratio enlargement (CDR > 0.6), neuroretinal rim thinning | High Priority |
| `4` | **Hypertension** | Arteriolar narrowing, arteriovenous (AV) nicking, copper/silver wiring | Moderate Urgency |
| `5` | **Myopia (Pathological Myopia)** | Tessellated fundus, peripapillary atrophy, posterior staphyloma | Routine Follow-up |
| `6` | **Normal** | Clear optic disc, intact macula, normal vascular caliber, no lesions | Routine |
| `7` | **Others** | Epiretinal membrane, retinal vein/artery occlusion, coloboma, other findings | Case-by-case |

---

## 📂 Repository Structure

```
OcularDiseaseRecognition/
│
├── Resnet50.ipynb               # ⭐ PRIMARY PIPELINE: Training, class-weight smoothing, ResNet50 model export
│
├── VGG16.ipynb                  # Comparative benchmark experiments with VGG16 architecture
├── ODIR_balanced.ipynb          # Balanced undersampling & class distribution experiments
├── odir_resize.ipynb            # Retinal image resizing and dimension normalization pipeline
├── odir_crop_images.ipynb       # Black border cropping and retinal circle region-of-interest extraction
├── odir_enhancement.ipynb       # Contrast enhancement (CLAHE, Gaussian filtering, MIRNet restoration)
│
├── app.py                       # Production FastAPI backend (REST API, Grad-CAM, Web UI server)
├── predict.py                   # Standalone Python CLI script for single-image diagnosis
├── batch_test.py                # Batch evaluation script for testing sets of images
├── class_indices.json           # JSON mapping of class IDs (0-7) to full diagnostic names
├── odir.csv                     # Dataset ground-truth annotations and patient metadata
│
├── models/
│   ├── best_resnet50_model.keras # Exported best ResNet50 model weights
│
├── static/                      # Web application static assets (CSS, JS, sample images)
│   ├── css/style.css            # Medical UI glassmorphic stylesheet
│   ├── js/app.js                # Frontend client logic & drag-and-drop file handler
│   ├── samples/                 # Sample clinical fundus images for one-click testing
│   └── checkpoints/             # Saliency checkpoint icons (macula, disc, vessels, periphery)
│
├── templates/
│   └── index.html               # Clinical decision support interface template
│
└── Testimages/                  # Validation fundus images for local testing
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Environment Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/sam190905/OcuscanAI.git
cd OcuscanAI

# Create virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install required packages
pip install tensorflow numpy pandas opencv-python pillow scikit-learn matplotlib fastapi uvicorn pydantic
```

---

### 2. Training the Model (`Resnet50.ipynb`)

To train or reproduce the production ResNet-50 weights:

1. Open **`Resnet50.ipynb`** in Jupyter Lab or Google Colab:
   ```bash
   jupyter notebook Resnet50.ipynb
   ```
2. Set the dataset directory path containing the categorized fundus folders (`Normal/`, `Glaucoma/`, `Diabetes/`, etc.).
3. Run all cells in sequence:
   * **Cell 2**: Preprocessing and real-time image augmentation (`ImageDataGenerator` with 20° rotation, shifts, and horizontal flips).
   * **Cell 3 & 5**: Automated calculation of square-root smoothed class weights.
   * **Cell 4**: Construction of the ResNet50 transfer learning graph.
   * **Cell 5**: Model training with `ModelCheckpoint` and `EarlyStopping` monitoring `val_auc`.
4. The trained model is automatically saved to `models/best_resnet50_model.keras`.

---

### 3. Launching the Clinical Web Application (`app.py`)

Run the interactive FastAPI server:

```bash
python app.py
```
*Alternatively, using Uvicorn directly:*
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

* **Web Interface**: Navigate to `http://localhost:8000` in your browser.
* **Interactive API Documentation**: Explore and test API endpoints at `http://localhost:8000/docs` (Swagger UI) or `http://localhost:8000/redoc`.

---

### 4. Running Command-Line Predictions (`predict.py`)

To run inference on an individual retinal image:

```bash
python predict.py "Testimages/Myopia.png"
```

**Example Output:**
```text
Loading model from models/best_resnet50_model.keras ...
Model loaded successfully!

============================================================
Image: Testimages/Myopia.png
Top Diagnosis: Myopia
Confidence: 94.62%
------------------------------------------------------------
Detailed Probability Distribution:
  - Myopia                               : 94.62%
  - Normal                               : 3.14%
  - Others                               : 1.25%
============================================================
```

---

### 5. Running Batch Validation (`batch_test.py`)

To run bulk evaluation across test directories:

```bash
python batch_test.py
```

---

## 🌐 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Renders the OccuScan AI clinical dashboard |
| `/predict` | `POST` | Upload single fundus image; returns top predictions, confidence, Grad-CAM heatmap, and clinical guidance |
| `/batch_predict` | `POST` | Multi-file upload for rapid screening of multiple retinal images |
| `/classes` | `GET` | Returns list of supported ocular disease classes |
| `/health` | `GET` | Health check endpoint returning model load status and GPU/CPU device state |

---

## 🔬 Explainable AI (Grad-CAM Integration)

In clinical triage, model interpretability is paramount. The platform implements **Gradient-weighted Class Activation Mapping (Grad-CAM)**:

1. Computes gradients of the target class score with respect to the final convolutional feature maps of ResNet50 (`conv5_block3_out`).
2. Generates a coarse localization map highlighting discriminative regions of interest.
3. Overlays the heatmap onto the original fundus photograph, validating whether the model attended to pathological hallmarks (such as cupping in Glaucoma, exudates in Diabetic Retinopathy, or drusen in AMD).

---

## 📊 Comparative Performance

| Architecture | Enhancement / Balancing | Top Metric Monitored | Validation AUC |
|---|---|---|---|
| **ResNet50 (Primary)** | **Resnet50.ipynb (Smoothed Class Weights)** | **`val_auc`** | **0.94+** |
| VGG16 | VGG16.ipynb | `val_loss` | ~0.91 |
| ResNet50 Baseline | Unweighted Flattening | `val_accuracy` | ~0.86 |

*The primary `Resnet50.ipynb` pipeline consistently achieves the highest multi-class discriminative performance and convergence stability.*

---


