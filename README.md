# Sarcasm Detection System for Indian Political Headlines

This project is an end-to-end sarcasm detection system designed for Indian political headlines using Machine Learning, Deep Learning, and Transformer-based NLP models.

The system performs:
- Web scraping of political headlines
- AI-assisted dataset annotation
- Text preprocessing
- Model training and evaluation
- Real-time sarcasm prediction through a Django web application

The project supports multiple prediction models:
- Machine Learning (TF-IDF + Linear SVM)
- CNN + LSTM Deep Learning Model
- Transformer-Based Model (RoBERTa)

---

# Project Structure

```bash
project_root/
│
├── Data/
│   ├── webscraping.py
│   ├── Labeling.py
│   ├── dataset.csv
│   └── balanced_dataset_12k.csv
│
├── Model_Training/
│   ├── ML Models
│   ├── CNN_LSTM Models
│   └── Transformer Models
│
├── models_store/
│   ├── lsvm_tfidf_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── tokenizer.pkl
│   └── cnn_lstm_enhanced.keras
│
├── sarcasm_RoBERTa_model/
│
├── sarcastic_app/
│   ├── backend/
│   │   ├── model_loader.py
│   │   ├── views.py
│   │   ├── urls.py
│   │   └── templates/
│   │       └── sarcastic_detection.html
│
├── manage.py
└── README.md
````

---

# Features

* Automated political headline scraping
* AI-generated sarcasm annotation pipeline
* Multiple NLP model implementations
* REST API for prediction
* Interactive Django frontend
* Real-time sarcasm classification
* Confidence score generation
* Multi-model comparison mode

---

# Technologies Used

## Backend

* Python
* Django 5

## Machine Learning

* Scikit-learn
* TensorFlow / Keras
* HuggingFace Transformers
* PyTorch

## NLP & Data Processing

* NLTK
* Pandas
* NumPy

## Frontend

* HTML
* JavaScript
* Chart.js

---

# Dataset Pipeline

## 1. Web Scraping

Political headlines are scraped from:

* The Hindu
* Indian Express

The scraper extracts headlines from:

* JSON-LD metadata
* ItemList structures

Output:

```bash
dataset.csv
```

---

## 2. AI-Assisted Data Annotation

The dataset is automatically labeled using AI prompting techniques.

Labels:

* sarcastic
* non-sarcastic

Generated fields:

* label
* confidence score

Output:

```bash
balanced_dataset_12k.csv
```

---

# Models Implemented

## Machine Learning Model

* TF-IDF Vectorization
* Linear SVM Classifier

## Deep Learning Model

* CNN + LSTM Hybrid Network

## Transformer Model

* RoBERTa-based Sequence Classification

---

# API Endpoint

## Predict Sarcasm

### Endpoint

```bash
POST /predict/
```

### Request Body

```json
{
  "headlines": [
    "Government announces miracle budget after protests"
  ],
  "model": "ALL"
}
```

### Available Model Options

* ML
* CNN_LSTM
* Transformer
* ALL

---

# Execution Instructions

## 1. Clone the Repository

```bash
git clone <repository_url>
cd <project_folder>
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install manually:

```bash
pip install django pandas numpy scikit-learn tensorflow torch transformers nltk keras-tuner beautifulsoup4 requests chartjs
```

---

## 4. Download NLTK Resources

Run Python shell:

```bash
python
```

Then execute:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

---

## 5. Ensure Model Files Exist

Place the following files inside `models_store/`:

```bash
lsvm_tfidf_model.pkl
tfidf_vectorizer.pkl
tokenizer.pkl
cnn_lstm_enhanced.keras
```

Also ensure:

```bash
sarcasm_RoBERTa_model/
```

exists in the project root.

---

## 6. Run the Django Server

Start the web application using:

```bash
python manage.py runserver
```

Server will run at:

```bash
http://127.0.0.1:8000/
```

---

# How to Use

1. Open the browser.
2. Navigate to:

```bash
http://127.0.0.1:8000/
```

3. Enter one or more political headlines.
4. Select the prediction model.
5. Click Analyze.
6. View:

   * Prediction result
   * Confidence score
   * Graph visualization

---

# AI Generated Technical Documentation

The project also contains AI-generated technical documentation explaining:

* System architecture
* Data flow
* Model pipeline
* Backend serving flow
* API handling
* Client-side aggregation
* Runtime inference

Documentation reference: 

The documentation includes:

* Architecture diagrams
* Sequence flows
* Module descriptions
* Runtime behavior
* Error handling
* Dependency breakdown

---

# Important Notes

* ALL mode performs prediction using all three models.
* Final aggregation logic is handled on the frontend.
* Transformer inference uses RoBERTa.
* Models are loaded during Django startup.
* Large models may require significant RAM and GPU resources.

---

# Future Improvements

* Add multilingual support
* Deploy using Docker
* Add user authentication
* Add database logging
* Real-time news feed integration
* Improve sarcasm explainability

---

# Author

Major Project – Sarcasm Detection in Indian Political Headlines

```

Technical documentation and architecture details referenced from uploaded project documentation. :contentReference[oaicite:1]{index=1}
```
