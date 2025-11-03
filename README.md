# 🎙️ Deepfake Audio Detective: ML/DL Model Showdown (Major Project II)

| Project Status | Version | License |
| :---: | :---: | :---: |
| **Complete** | `v1.0.0` | [MIT](LICENSE) |

<br>

## 🚀 Project Overview: Spotting the Fakes

**Deepfakes are getting way too real!** 😬 This project dives deep into the world of **Audio Deepfake Detection**. Our goal was to build and rigorously compare multiple Machine Learning (ML) and Deep Learning (DL) models to find the ultimate champion for spotting synthesized or manipulated voice recordings (a.k.a. "spoofing" attacks).

This repo is the culmination of our Major Project II, focusing on robustness, accuracy, and detailed comparative analysis.

---

## ✨ Features & Model Showcase

We implemented and trained a whole squad of models to see which one has the best detective skills. You can find all the code for these approaches organized neatly in their respective folders:

### 🧠 Deep Learning Architectures
These models are flexing some serious computational power, primarily using spectrogram analysis (audio converted to images) for classification:

* **ResNet50:** A classic, known for its deep residual connections.
* **DenseNet121:** Connects every layer to every other layer in a feed-forward fashion.
* **EfficientNetB0:** Scaled efficiently for great performance with fewer parameters.
* **ConvNeXt-Tiny:** A modern CNN inspired by Vision Transformers.
* **MobileNetV3:** Lightweight model optimized for mobile and edge devices (but still crushing it here!).

### 📊 Traditional Machine Learning
We also brought in some ML classics to set a strong baseline and compare against the DL heavyweights:

* **Support Vector Machine (SVM)**
* **Random Forest**
* **K-Nearest Neighbors (KNN)**
* **Decision Tree**
* **Naive Bayes**

---

## 🛠️ Tech Stack & Dependencies

Built by engineers, for engineers! This project is 100% Python-powered.

* **Language:** Python 🐍
* **Deep Learning:** PyTorch / TensorFlow (or a similar framework like Keras)
* **ML & Data:** Scikit-learn, Pandas, NumPy
* **Audio Processing:** Librosa or Torchaudio
* **Visualization:** Matplotlib, Seaborn

### ⚙️ Installation

Ready to run this thing? You'll need Python 3.x installed.

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Infinite-FE/Major-Project-II-01CE0807-.git](https://github.com/Infinite-FE/Major-Project-II-01CE0807-.git)
    cd Major-Project-II-01CE0807-
    ```

2.  **Create & Activate Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt 
    # (Assuming you have a requirements.txt file with all dependencies!)
    ```

---

## 🏃 Getting Started & Usage

### 1. Data Preparation
*(**Heads up:** You'll need to link your specific dataset here or provide instructions on how to download it, e.g., ASVspoof 2019.)*

Place your audio files (or processed features) and label files in a designated directory, following the structure expected by the model scripts.

### 2. Training a Model

To train any specific model (e.g., the ResNet50 classifier):

```bash
# Example: Run the training script inside the ResNet50 folder
python ResNet50/train.py --data_path ./path/to/your/dataset
