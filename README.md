# Email Spam Classification Web App

An end-to-end Machine Learning project that classifies emails as Spam or Ham (Legitimate). This project features core algorithms implemented from scratch and a functional web interface for real-time testing.

## 📌 Overview

This project was developed for the Introduction to Artificial Intelligence course (Group G29). We implemented a complete data science pipeline—from raw data preprocessing to model deployment—to solve the problem of unsolicited email detection.

**Key Highlight:** Unlike standard implementations using high-level libraries, we built the **Naive Bayes** and **K-Nearest Neighbors (KNN)** algorithms **from scratch** to demonstrate a deep understanding of the underlying mathematical principles.

## 📊 Dataset

* **Source**: [Email Spam Classification Dataset](https://www.google.com/url?q=https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Fbalaka18%2Femail-spam-classification-dataset-csv%2Fdata)

* **Features**: 3000 word-frequency columns derived from thousands of emails.

*   **Processing:**
    *   Removed 541 duplicate records to ensure data integrity.
    *   Addressed class imbalance (originally ~2:1 Ham to Spam ratio) using **Random Under-Sampling** to create a balanced training set of 935 samples per class.

## 🛠️ Methodology

### 1. Custom Algorithm Implementation

* **Naive Bayes:**

    * Implemented with Laplace Smoothing ($1.0$) to prevent zero-probability errors.

    * Used Log-Likelihood summation to avoid numerical underflow during probability multiplication.

* **K-Nearest Neighbors (KNN):**

    * Developed with support for Euclidean, Manhattan, and Cosine Similarity metrics.

    * Hyperparameter $k$ was optimized through validation testing (Best $k=1$ with Cosine metric).

### **2. Web Application**

  * **Frontend:** Interactive dashboard built with Streamlit.

  * **Features:** Users can input custom email text, select between algorithms, and receive an instant classification result.

## 📈 Performance Analysis

The models were evaluated on an unseen test set ($n=927$).

The models were evaluated on an unseen test set ($n = 927$).

| Metric | Naive Bayes (Custom) | KNN ($k = 1$, Cosine) |
| :--- | :--- | :--- |
| **Accuracy** | **95.14%** | **84.46%** |
| **Precision (Spam)** | 0.89 | 0.70 |
| **Recall (Spam)** | 0.97 | 0.90 |
| **F1-Score (Macro)** | **0.95** | **0.83** |

**Conclusion:** Naive Bayes significantly outperformed KNN in accuracy, resource efficiency, and prediction balance. KNN showed a tendency to "catch all" spam (high recall) but suffered from a high false-positive rate (low precision).

## 📂 Project Structure

```text
├── app.py              # Streamlit application source code
├── nb_model.joblib     # Pre-trained Naive Bayes model
├── knn_model.joblib    # Pre-trained KNN model
├── requirements.txt    # Project dependencies
└── README.md           # Documentation
```

### 🚀 Installation & Usage

1. **Clone the repository:** 
   Truy cập vào [GitHub Repository](https://github.com/dangminhquandk/Email_Spam_Classification.git)
2. **Install dependencies:**

```text
pip install -r requirements.txt
```

3. **Run the App:**
```text
streamlit run app.py
```

## 👥 Contributors (Group G29)

**Đặng Minh Quân** - 202416323 (Naive Bayes Research, Model Evaluation)

**Lê Quang Phúc** - 202416316 (Data Preprocessing, KNN Algorithm)

**Hồ Sỹ Toà**n - 202416368 (Data Acquisition & Cleaning)

**Hoàng Đặng Xuân Mỹ** - 202416295 (Comparative Analysis & Reporting)
