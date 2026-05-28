import streamlit as st
import numpy as np
import pandas as pd
import joblib
import re
from collections import Counter

class Naive_Bayes:
    def __init__(self, laplace_smoothing=1.0):
        # Hệ số làm trơn để tránh lỗi nhân cho 0
        self.laplace_smoothing = laplace_smoothing

        # Lưu xác suất tiên nghiệm P(c) - VD: P(Spam), P(Ham)
        self.prior_probs = {}

        # Lưu log xác suất có điều kiện log P(w|c) cho từng từ trong từng lớp
        self.log_likelihood = {}

        # Danh sách các nhãn [0, 1]
        self.classes = []

    def fit(self, X, y):
        n_samples, n_features = X.shape # Số dòng (email), Số cột (từ)
        self.classes = np.unique(y)     # Tìm các nhãn duy nhất: [0, 1]

        for c in self.classes:
            # Lấy tất cả các dòng email thuộc nhãn c (VD: Lấy hết Spam)
            X_c = X[y == c]

            # P(c) = Số lượng email lớp c / Tổng số email
            self.prior_probs[c] = X_c.shape[0] / n_samples

            # Tổng số lần xuất hiện của TỪNG từ trong toàn bộ email lớp c
            # word_counts: Mảng 1 chiều chứa tổng từng cột
            word_counts = np.sum(X_c, axis=0) + self.laplace_smoothing

            # Tổng số từ (tất cả các từ) trong lớp c + phần bù Laplace
            total_counts = np.sum(word_counts)

            # Tính Log xác suất để tránh số quá nhỏ (Underflow)
            # log( P(w|c) )
            self.log_likelihood[c] = np.log(word_counts / total_counts)

    def predict(self, X):
        result = [self._predict_single(x) for x in X]
        return np.array(result)

    def _predict_single(self, x):
        best_posterior = -np.inf
        best_class = None

        for c in self.classes:
            # log P(c|x) ∝ log P(c) + Σ (tần_suất_từ * log P(từ|c))

            # Lấy Log Tiên nghiệm (log P(c))
            log_prior = np.log(self.prior_probs[c])

            # Phép nhân vô hướng (Dot product) giữa tần suất từ trong x và log xác suất đã học
            # x * log_likelihood tương đương với việc cộng dồn log xác suất của từng từ
            log_likelihood_sum = np.dot(x, self.log_likelihood[c])

            # Tính Log Hậu nghiệm
            posterior = log_prior + log_likelihood_sum

            # Tìm lớp có xác suất lớn nhất
            if posterior > best_posterior:
                best_posterior = posterior
                best_class = c

        return best_class

def cosine(x1, x2):
    dot_product = np.dot(x1, x2)
    norm_x1 = np.linalg.norm(x1)
    norm_x2 = np.linalg.norm(x2)
    # 1 - cosine similarity = distance
    return 1 - (dot_product / (norm_x1 * norm_x2 + 1e-9))

def euclidean(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2)**2))
    return distance

def manhattan(x1, x2):
    distance = np.sum(np.abs(x1 - x2))
    return distance

class KNN:
    def __init__(self, k=5, metric='l1'):
        self.k = k
        self.metric = metric

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # select the distance function based on self.metric
        if self.metric == 'cosine':
            dist_func = cosine
        elif self.metric == 'l2':
            dist_func = euclidean
        else:
            dist_func = manhattan

        # compute the distance
        distances = [dist_func(x, x_train) for x_train in self.X_train]

        # get the closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # majority voye
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]

@st.cache_data
def load_vocabulary():
    df = pd.read_csv('/content/drive/MyDrive/emails.csv')
    # Get all column names except 'Email No.' and 'Prediction'
    vocabulary = [col for col in df.columns if col not in ['Email No.', 'Prediction']]
    return vocabulary

@st.cache_resource
def load_models():
    # Load model
    nb = joblib.load('nb_model.joblib')
    # knn = joblib.load('knn_model.joblib')
    # return nb, knn
    return nb

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Extract words (alphanumeric sequences)
    words = re.findall(r'\b[a-z]+\b', text)
    return words

def text_to_feature_vector(text, vocabulary):
    # Preprocess text to get words
    words = preprocess_text(text)

    # Count word frequencies
    word_counts = Counter(words)

    # Create feature vector
    feature_vector = []
    for word in vocabulary:
        feature_vector.append(word_counts.get(word, 0))

    return np.array(feature_vector)

def main():
    models = load_models()
    vocabulary = load_vocabulary()

    st.set_page_config(page_title="Spam Classifier", layout="centered")

    st.title("Email Spam Classifier")
    st.write("---")

    try:
        vocabulary = load_vocabulary()
        nb_model = load_models()
    except FileNotFoundError:
        st.error("Thiếu file cấu hình (`emails.csv` hoặc `nb_model.joblib`). Hãy kiểm tra lại repository.")
        st.stop()
    
    # Model selector
    algorithm = st.selectbox(
        "Select Algorithm:",
        ("Naive Bayes", "K-Nearest Neighbors")
    )

    st.write("---")

    # Input area
    st.subheader("Enter Email to Check")
    email_input = st.text_area("Email:", height=150, placeholder="Example: Congratulations! You won a prize...")

    # Button
    if st.button("Classify Now", type="primary"):
        if not email_input:
            st.warning("Please enter email before checking.")
        else:
            # Select model
            if algorithm == "Naive Bayes":
                model = models[0]
            else:
                model = models[1]

            # Preprocess input text to feature vector
            feature_vector = text_to_feature_vector(email_input, vocabulary)

            # Prediction
            prediction = model.predict([feature_vector])[0]

            # Result
            st.write("---")
            st.subheader("Result:")

            if prediction == 1:
                st.error("THIS IS SPAM EMAIL")
            else:
                st.success("THIS IS LEGITIMATE EMAIL")

if __name__ == '__main__':
    main()
