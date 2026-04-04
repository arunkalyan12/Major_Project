import pickle

import joblib
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
import re,unicodedata


# Load tokenizer
with open("models_store/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load CNN + LSTM
import os
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cnn_lstm_model = load_model(
    os.path.join(BASE_DIR, "models_store", "cnn_lstm_enhanced.keras")
)

# 🔥 ML MODEL + TFIDF
ml_model = joblib.load(
    os.path.join(BASE_DIR, "models_store", "lsvm_tfidf_model.pkl")
)

tfidf_vectorizer = joblib.load(
    os.path.join(BASE_DIR, "models_store", "tfidf_vectorizer.pkl")
)
# Load Transformer Model
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

transformer_path = os.path.join(BASE_DIR, "sarcasm_RoBERTa_model")
transformer_tokenizer = AutoTokenizer.from_pretrained(transformer_path)
transformer_model = AutoModelForSequenceClassification.from_pretrained(transformer_path)


# Text Cleaning
def clean_text(t):
    t = unicodedata.normalize("NFC", str(t))
    t = re.sub(r"[\u200B-\u200D\uFEFF]", "", t)
    t = re.sub(r"x[0-9A-Fa-f]{4}", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t
