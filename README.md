
# NeuroScan AI: Precision Brain Tumor Insight 🧠

![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![TensorFlow Version](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit App](https://img.shields.io/badge/Streamlit-App-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌟 Introduction

Welcome to **NeuroScan AI: Precision Brain Tumor Insight** – an advanced deep learning project focused on the automated classification of brain MRI images. This tool is designed to assist medical professionals by accurately identifying four critical categories: **Glioma Tumor, Meningioma Tumor, No Tumor, and Pituitary Tumor**. Leveraging robust deep learning models, this project aims to enhance diagnostic efficiency and precision in medical image analysis.

## ✨ Features

* **Multi-Class Classification:** Classifies brain MRI scans into 4 distinct categories.
* **Custom CNN Model:** Utilizes a highly accurate custom Convolutional Neural Network.
* **Transfer Learning Exploration:** Explored pre-trained models (MobileNetV2) for feature extraction and fine-tuning.
* **Comprehensive Evaluation:** Rigorous model evaluation using accuracy, precision, recall, F1-score, and confusion matrices.
* **Interactive Streamlit Web App:** A user-friendly interface for real-time MRI image classification.
* **Data Augmentation:** Robust augmentation pipeline to improve model generalization.
* **Responsible AI Practices:** Includes ethical disclaimers for medical applications.

## 🚀 Models & Performance

The project meticulously trained and evaluated multiple deep learning models. The **Custom CNN Model** emerged as the top performer, demonstrating superior accuracy and balanced performance on unseen data.

| Model                                    | Test Accuracy |
| :--------------------------------------- | :------------ |
| **Custom CNN Model** | **0.9146** |
| Transfer Learning (Feature Extraction)   | 0.8211        |
| Transfer Learning (Fine-Tuning)          | 0.2642        |

*(Detailed evaluation reports and confusion matrices are available in the project report.)*

## 🛠️ Installation & Setup

To get this project up and running locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/NeuroScan-AI.git](https://github.com/YourUsername/NeuroScan-AI.git)
    cd NeuroScan-AI
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    *(You will need to create a `requirements.txt` file by running `pip freeze > requirements.txt` in your Colab environment or manually listing `tensorflow`, `streamlit`, `numpy`, `matplotlib`, `Pillow`, `opencv-python`)*

4.  **Download the trained model:**
    The best-performing model (`best_custom_cnn_model.keras`) is required.
    * Download it from [Google Drive Link/Kaggle Dataset Link] (if hosted separately)
    * **OR:** Place your saved `best_custom_cnn_model.keras` file (from your Colab `saved_models/` directory) directly into the project's root folder where `streamlit_app.py` resides.

## 🏃 Usage

To run the Streamlit web application:

```bash
streamlit run streamlit_app.py
````

This will open the application in your default web browser (usually `http://localhost:8501`). You can then upload an MRI image and receive a real-time classification.

## 📂 Project Structure

```
NeuroScan-AI/
├── streamlit_app.py           # The Streamlit web application
├── best_custom_cnn_model.keras # Your best trained model (downloaded)
├── data/                      # (Optional) Directory for raw/processed data if included
│   ├── train/
│   ├── valid/
│   └── test/
├── notebooks/                 # (Optional) Jupyter/Colab notebooks for development
│   ├── Data_Preparation.ipynb
│   ├── Model_Training.ipynb
│   └── Model_Evaluation.ipynb
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 📈 Future Work

  * **Uncertainty Quantification:** Implement Monte Carlo Dropout to provide uncertainty estimates with predictions.
  * **Explainable AI (XAI):** Integrate techniques like Grad-CAM for visual explanations of model decisions.
  * **Dataset Expansion:** Incorporate larger and more diverse MRI datasets, including different sequences.
  * **Advanced Architectures:** Explore Vision Transformers (ViT) and more sophisticated CNNs.
  * **Tumor Segmentation:** Extend the project to include precise tumor segmentation.
  * **Clinical Validation:** Conduct studies with medical professionals for real-world validation.

## 🤝 Contributing

Contributions are welcome\! If you have suggestions or improvements, please open an issue or submit a pull request.


## 📧 Contact

For any questions or collaborations, feel free to reach out:

  * Arya Jain: mailin2.aryajain@gmail.com

## ⚠️ Disclaimer

**This application is for educational and demonstrative purposes only and should not be used for medical diagnosis. Always consult with a qualified healthcare professional for any medical concerns.**

```
```
