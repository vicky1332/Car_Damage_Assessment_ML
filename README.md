# Car Damage Assessment Using Computer Vision and Machine Learning

A classical computer-vision and machine-learning project that identifies multiple visible car-damage categories from images and analyzes annotation-derived visible damage extent using the CarDD dataset.

## Project Overview

This project uses the **CarDD: A New Dataset for Vision-Based Car Damage Detection** dataset and a classical computer-vision and machine-learning workflow to identify visible car-damage categories from images.

The system uses **OpenCV**, **HOG**, **LBP**, **StandardScaler**, **PCA**, and a **One-vs-Rest Linear SVM** for multi-label damage classification.

The project follows the workflow:

**Image Preprocessing → HOG + LBP Feature Extraction → Standardization → PCA → One-vs-Rest Linear SVM → Multi-Label Damage Classification → Damage Extent Analysis → Streamlit Deployment**

The final model identifies one or more of the following damage categories:

- Dent
- Scratch
- Crack
- Glass Shatter
- Lamp Broken
- Tire Flat

The project also uses CarDD segmentation annotations to analyze the **visible extent of damage**. This analysis is based on annotated damage-area ratios and should not be interpreted as mechanical, structural, or safety severity.

## Dataset

**Dataset:** CarDD — A New Dataset for Vision-Based Car Damage Detection

CarDD contains **4,000 high-resolution car images** and more than **9,000 annotated damage instances** across six damage categories.

The official dataset split is retained:

| Split | Images |
|---|---:|
| Training | 2,816 |
| Validation | 810 |
| Test | 374 |
| Total | 4,000 |

Because a single image can contain multiple damage categories, the problem is formulated as a **multi-label classification task**.

The dataset provides COCO-style annotations containing damage categories, bounding boxes, segmentation information, and annotated damage area.

Dataset source:

https://cardd-ustc.github.io/

## Computer Vision and Machine Learning Workflow

### Image Preprocessing

Images are processed using OpenCV:

- Convert images to grayscale
- Resize to a consistent resolution
- Apply Gaussian smoothing
- Normalize pixel values

### Feature Extraction

Two complementary handcrafted descriptors are used:

**HOG — Histogram of Oriented Gradients**

Captures edge and shape information from the image.

**LBP — Local Binary Pattern**

Captures local texture information.

The HOG and LBP descriptors are concatenated into a single feature vector.

### Feature Standardization

`StandardScaler` is fitted using the training features and then applied to the validation and test features.

### PCA Dimensionality Reduction

Principal Component Analysis is used to reduce the dimensionality of the feature space while retaining approximately **95% of the variance**.

The original feature representation contains **8,110 features**, which is reduced to **1,335 PCA components**.

### Multi-Label Classification

A **One-vs-Rest Linear SVM** is used for multi-label classification.

One binary classifier is trained for each damage category, allowing a single image to contain multiple predicted damage types.

Class imbalance is handled using balanced class weights.

The regularization parameter is selected using the validation set, with **Micro F1-score** as the primary selection metric.

The selected value is:

```text
C = 0.01
```

## Model Evaluation

The final model is evaluated using:

- Hamming Loss
- Micro Precision
- Micro Recall
- Micro F1-score
- Macro F1-score

### Validation Performance

| Metric | Result |
|---|---:|
| Hamming Loss | 0.2576 |
| Micro Precision | 0.4944 |
| Micro Recall | 0.5368 |
| Micro F1-score | 0.5147 |
| Macro F1-score | 0.4835 |

### Final Test Performance

The held-out test set is used only once for the final evaluation.

| Metric | Result |
|---|---:|
| Hamming Loss | 0.2834 |
| Micro Precision | 0.4410 |
| Micro Recall | 0.5459 |
| Micro F1-score | 0.4879 |
| Macro F1-score | 0.4549 |

## Damage Extent Analysis

The CarDD segmentation annotations are additionally used to analyze the visible extent of damage.

For each annotated damage instance, the damage-area ratio is calculated as:

**Damage Area Ratio = Annotated Damage Area / Image Area**

The training-set distribution of these ratios is used to define three visible-extent bands:

- Minor
- Moderate
- Severe

The thresholds are based on the 33rd and 67th percentiles of the training annotation distribution.

These categories describe **visible damage extent** and should not be interpreted as mechanical, structural, or safety severity.

### Important Deployment Limitation

The deployed classical classifier performs **image-level damage classification**. It does not directly perform pixel-level segmentation on a newly uploaded image.

Therefore, the Streamlit application does not claim to measure the exact percentage of visible damage in a new uploaded image.

The annotation-derived Minor, Moderate, and Severe bands are used for analysis of the CarDD training annotations and are not presented as exact severity measurements for new uploaded images.

## Streamlit Application

The project includes an interactive Streamlit application for image-based car damage assessment.

The application provides:

- Car image upload
- Image preview
- Multi-label damage detection
- No-clear-damage handling
- Project methodology
- Model limitations
- A modern automotive-themed user interface

The application uses the saved trained model artifact and does **not** retrain the model at runtime.

### Application Workflow

```text
Uploaded Car Image
        ↓
OpenCV Preprocessing
        ↓
HOG + LBP Feature Extraction
        ↓
StandardScaler
        ↓
PCA
        ↓
One-vs-Rest Linear SVM
        ↓
Detected Damage Categories
```

## No Clear Damage Case

The CarDD training dataset does not contain a dedicated `no_damage` class.

Therefore, if the classifier does not return any supported damage category, the application displays:

**No Clear Damage Detected**

This does not mean that the vehicle is guaranteed to be damage-free.

It means that the trained classifier did not identify any of the supported CarDD damage categories in the uploaded image.

## Model Artifact

The trained preprocessing and classification components are packaged into a single scikit-learn pipeline:

```text
StandardScaler
      ↓
PCA
      ↓
One-vs-Rest Linear SVM
```

The custom HOG + LBP feature extraction remains outside the scikit-learn pipeline because it is implemented as a custom image-processing step.

The saved model artifact is:

```text
car_damage_model.pkl
```

The artifact contains:

```text
pipeline
category_columns
```

This allows the Streamlit application to load the same fitted preprocessing and classification pipeline used during model development.

## Project Structure

```text
Car-Damage-Assessment-Classical-ML/
│
├── Car_Damage_Assessment_Classical_ML_GitHub_Final.ipynb
├── app.py
├── model_utils.py
├── car_damage_model.pkl
├── requirements.txt
├── README.md
│
├── .streamlit/
│   └── config.toml
│
└── assets/
    └── ...
```

The complete CarDD dataset is **not required at runtime** by the Streamlit application.

Only the trained model artifact and application dependencies are required for inference.

## How to Run Locally

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Car-Damage-Assessment-Classical-ML
```

### 2. Install the required packages

```bash
pip install -r requirements.txt
```

### 3. Make sure the model artifact is present

The following file must be available in the project directory:

```text
car_damage_model.pkl
```

### 4. Start the Streamlit application

```bash
streamlit run app.py
```

The application will be available at:

```text
https://cardamageassessmentml-mibdkyqpxcihxmd6uterj5.streamlit.app/
```

## Deployment

The Streamlit application can be deployed using **Streamlit Community Cloud**.

The GitHub repository should contain:

```text
app.py
model_utils.py
car_damage_model.pkl
requirements.txt
.streamlit/
```

The complete CarDD dataset is not required for deployment because the application performs inference using the saved model artifact.

## Important Limitations

This project should be interpreted as a **classical computer-vision and machine-learning portfolio prototype**.

Important limitations include:

- The classifier performs image-level multi-label damage classification.
- The current deployed classifier does not localize damage regions in new uploaded images.
- Visible damage extent analysis is based on CarDD segmentation annotations.
- Minor, Moderate, and Severe describe visible damage extent and should not be interpreted as mechanical, structural, or safety severity.
- The CarDD dataset contains no dedicated `no_damage` class.
- A "No Clear Damage Detected" result does not guarantee that a vehicle is damage-free.
- The system is intended as an educational portfolio project and should not be used as a substitute for professional vehicle inspection or safety assessment.

## References

1. Wang, X., Li, W., & Wu, Z. (2023). **CarDD: A New Dataset for Vision-Based Car Damage Detection.** *IEEE Transactions on Intelligent Transportation Systems, 24(7), 7202–7214.* DOI: 10.1109/TITS.2023.3258480.

2. CarDD — Official Dataset Website  
   https://cardd-ustc.github.io/

3. Dalal, N., & Triggs, B. (2005). **Histograms of Oriented Gradients for Human Detection.**

4. Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002). **Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns.**

5. scikit-learn Documentation  
   https://scikit-learn.org/

6. OpenCV Documentation  
   https://docs.opencv.org/

7. scikit-image Documentation  
   https://scikit-image.org/

## Author

**Trivikram Kambhampati**
