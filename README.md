# Car Damage Assessment & Detection Using Computer Vision and Classical Machine Learning

An interactive, AI-assisted automotive vision application built with **Streamlit**, **OpenCV**, **scikit-image**, and **scikit-learn**. The system automatically classifies vehicle damage types from uploaded photographs across six primary categories using handcrafted feature descriptors and classical machine learning.

---

## 🚗 Overview & Problem Statement

Automated car damage detection plays a key role in intelligent transportation systems, automated insurance intake, and vehicle inspection workflows. 

This project implements an end-to-end classical computer vision and machine learning pipeline trained on the official **CarDD (Car Damage Dataset)** to perform multi-label damage classification across six primary categories:

- **Dent**
- **Scratch**
- **Crack**
- **Glass Shatter**
- **Lamp Broken**
- **Tire Flat**

---

## ⚙️ Machine Learning & Inference Architecture

The application uses the pre-trained model artifact (`car_damage_model.pkl`) generated during training:

```
Uploaded Car Image (JPG/PNG)
            │
            ▼
    OpenCV Preprocessing
  [Grayscale ➔ 128×128 ➔ GaussianBlur(3×3) ➔ Float32 Normalization [0,1]]
            │
            ▼
 Handcrafted Feature Extraction
  [HOG (8,100-D) + LBP Histogram (10-D) ➔ Concatenated 8,110-D Vector]
            │
            ▼
    StandardScaler (Fitted)
  [Standardizes 8,110 features using training mean and variance]
            │
            ▼
      PCA (Fitted)
  [Reduces dimension to 1,335 components retaining 95% variance]
            │
            ▼
 One-vs-Rest LinearSVC (Fitted)
  [Independent binary classifiers for multi-label prediction]
            │
            ▼
 Detected Damage Categories & Automated Extent Classification
```

---

## 📊 Dataset & Model Evaluation Metrics

### Dataset: CarDD (Car Damage Dataset)
- **Images:** 4,000 high-resolution car damage images
- **Damage Annotations:** 9,000+ COCO-style bounding box and segmentation records
- **Categories:** Dent, Scratch, Crack, Glass Shatter, Lamp Broken, Tire Flat

### Model Performance Metrics
| Metric | Validation Set | Test Set |
| :--- | :---: | :---: |
| **Micro F1-Score** | **0.515** | **0.488** |
| **Hamming Loss** | 0.224 | 0.236 |
| **Micro Precision** | 0.395 | 0.380 |
| **Micro Recall** | 0.741 | 0.682 |
| **Macro F1-Score** | 0.478 | 0.443 |

---

## 📁 Repository Structure

```
Car Damage Assessment & Detection/
│
├── app.py                                          # Main Streamlit web application
├── model_utils.py                                  # Preprocessing, feature extraction & model loading utilities
├── car_damage_model.pkl                            # Fitted sklearn Pipeline artifact (Scaler + PCA + LinearSVC)
├── Car_Damage_Assessment_&_Repair_Cost_Estimation.ipynb # Jupyter notebook documenting dataset & pipeline
├── requirements.txt                                # Python dependencies for local run & cloud deployment
├── README.md                                       # Project documentation & deployment guide
├── .streamlit/
│   └── config.toml                                 # Custom dark theme configuration
└── assets/
    ├── style.css                                   # Custom CSS stylesheet
    └── car_bg.jpg                                  # Transparent automotive blueprint background image
```

---

## 🚀 Quickstart & Local Installation

### Prerequisites
- Python 3.9 or higher
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/car-damage-assessment.git
cd car-damage-assessment
```

### 2. Create and Activate Virtual Environment
- **On Windows:**
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```
- **On macOS/Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Streamlit Application
```bash
streamlit run app.py
```
The app will open automatically in your browser at `http://localhost:8503`.

---

## ☁️ Deployment on Streamlit Community Cloud

1. Push your repository to **GitHub** (ensure `app.py`, `model_utils.py`, `car_damage_model.pkl`, `requirements.txt`, `.streamlit/`, and `assets/` are committed).
2. Log into [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Click **"New app"** and select your GitHub repository, branch (`main`), and main file path (`app.py`).
4. Click **"Deploy!"**.

---

## ⚠️ Methodological Notes

1. **Category Classification vs. Segmentation:** The deployed classifier predicts damage categories from handcrafted image features. It does *not* perform direct pixel-level segmentation on newly uploaded images.
2. **No-Clear-Damage State:** If the model returns zero positive damage categories, the UI displays *"No clear damage detected"*. This signifies that none of the 6 supported categories were identified, but does not guarantee the vehicle is completely damage-free.

---

## 📜 License & Acknowledgments

- **CarDD Dataset:** [CarDD USTC Project Page](https://cardd-ustc.github.io/)
- Built with [Streamlit](https://streamlit.io/), [OpenCV](https://opencv.org/), [scikit-image](https://scikit-image.org/), and [scikit-learn](https://scikit-learn.org/).
