# 🐦 Twitter Sentiment Analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rajaykumar12/Twitter_Sentimental_analysis/blob/main/Sentiment_analysis.ipynb)

A comprehensive machine learning project that performs binary sentiment classification on Twitter data using Natural Language Processing techniques and Logistic Regression. This model analyzes tweet text to determine whether the sentiment expressed is positive or negative.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Performance Metrics](#performance-metrics)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [Future Improvements](#future-improvements)
- [License](#license)

## 🎯 Overview

This project implements a complete sentiment analysis pipeline that:

- **Preprocesses** raw tweet data using advanced NLP techniques
- **Trains** a Logistic Regression classifier on 1.6 million labeled tweets
- **Evaluates** model performance using standard metrics
- **Saves** the trained model for future deployment and predictions

### Sentiment Classifications:
- **0**: Negative sentiment 😞
- **1**: Positive sentiment 😊

## 📊 Dataset

The project utilizes the [**Sentiment140 dataset**](https://www.kaggle.com/datasets/kazanova/sentiment140) from Kaggle, which contains:

- **1,600,000 tweets** with sentiment labels
- **6 features**: target, ids, date, flag, user, text
- **Balanced dataset** with equal positive and negative samples
- **Pre-labeled data** for supervised learning

### Dataset Structure:
```
Columns: [target, ids, date, flag, user, text]
Shape: (1,600,000, 6)
Target Distribution: 50% positive, 50% negative
```

## ✨ Features

### 🔧 Text Preprocessing Pipeline
- **Special Character Removal**: Cleans non-alphabetic characters
- **Case Normalization**: Converts all text to lowercase
- **Stop Words Removal**: Eliminates common English stop words
- **Stemming**: Reduces words to their root form using Porter Stemmer
- **Tokenization**: Splits text into meaningful tokens

### 🤖 Machine Learning Pipeline
- **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
- **Data Splitting**: 80% training, 20% testing with stratified sampling
- **Model Training**: Logistic Regression with optimized parameters
- **Model Persistence**: Saves trained model using pickle for deployment

## 🚀 Installation

### Prerequisites
- Python 3.7+
- Kaggle account and API credentials

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Rajaykumar12/Twitter_Sentimental_analysis.git
   cd Twitter_Sentimental_analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Kaggle API**
   ```bash
   # Download kaggle.json from your Kaggle account
   mkdir -p ~/.kaggle
   cp kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

### Required Packages
```python
pandas>=1.3.0
numpy>=1.21.0
nltk>=3.6
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
kaggle>=1.5.12
opendatasets>=0.1.20
```

## 📖 Usage

### Quick Start

1. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook Sentiment_analysis.ipynb
   ```

2. **Run all cells** to execute the complete pipeline:
   - Data download and preprocessing
   - Model training and evaluation
   - Model saving and testing

### Programmatic Usage

```python
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model
model = pickle.load(open('trained_model.sav', 'rb'))

# Load the vectorizer (you'll need to save this too)
vectorizer = pickle.load(open('vectorizer.sav', 'rb'))

# Preprocess and predict new text
def predict_sentiment(text):
    # Apply same preprocessing as training data
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)
    
    return "Positive" if prediction[0] == 1 else "Negative"

# Example usage
sentiment = predict_sentiment("I love this product!")
print(f"Sentiment: {sentiment}")
```

## 🏗️ Model Architecture

### Text Processing Flow
```
Raw Tweet → Clean Text → Tokenize → Remove Stopwords → Stem → TF-IDF → Features
```

### Model Specifications
- **Algorithm**: Logistic Regression
- **Solver**: Default (lbfgs)
- **Max Iterations**: 1000
- **Feature Extraction**: TF-IDF Vectorization
- **Input Dimension**: Variable (depends on vocabulary size)
- **Output**: Binary classification (0 or 1)

## 📈 Performance Metrics

| Metric | Training Set | Test Set |
|--------|-------------|----------|
| **Accuracy** | ~79.0% | ~79.0% |
| **Dataset Size** | 1,280,000 | 320,000 |

### Model Characteristics
- **No Overfitting**: Similar training and testing accuracy
- **Balanced Performance**: Works well on both positive and negative sentiments
- **Scalable**: Can handle large volumes of text data efficiently

## 📁 Project Structure

```
Twitter_Sentimental_analysis/
│
├── Sentiment_analysis.ipynb    # Main Jupyter notebook
├── README.md                   # Project documentation
├── requirements.txt           # Python dependencies
├── trained_model.sav         # Serialized trained model
├── kaggle.json              # Kaggle API credentials (not tracked)
│
└── sentiment140/            # Dataset directory (auto-created)
    └── training.1600000.processed.noemoticon.csv 
```

## 🔌 API Reference

### Core Functions

#### `stemming(content)`
Preprocesses text by removing special characters, converting to lowercase, removing stopwords, and applying stemming.

**Parameters:**
- `content` (str): Raw text to be processed

**Returns:**
- `str`: Preprocessed text ready for vectorization

#### Model Training Pipeline
1. **Data Loading**: `pd.read_csv()` with proper encoding
2. **Preprocessing**: Apply `stemming()` function to all tweets
3. **Vectorization**: `TfidfVectorizer().fit_transform()`
4. **Training**: `LogisticRegression().fit()`
5. **Evaluation**: `accuracy_score()`

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Areas
- Model performance improvements
- Additional preprocessing techniques
- Web interface development
- Real-time analysis features
- Documentation enhancements

## 🔮 Future Improvements

### Short Term
- [ ] **Save vectorizer** alongside the model for complete pipeline persistence
- [ ] **Add cross-validation** for more robust performance estimation
- [ ] **Implement confusion matrix** and detailed classification metrics
- [ ] **Create requirements.txt** with exact version specifications

### Medium Term
- [ ] **Advanced preprocessing**: Handle emojis, URLs, and mentions
- [ ] **Ensemble methods**: Combine multiple algorithms for better accuracy
- [ ] **Feature engineering**: Add sentiment lexicon features
- [ ] **Model comparison**: Test Random Forest, SVM, and Neural Networks

### Long Term
- [ ] **Deep Learning**: Implement LSTM/BERT for improved accuracy
- [ ] **Multi-class classification**: Extend to neutral, very positive/negative
- [ ] **Real-time API**: Deploy model as REST API service
- [ ] **Web interface**: Create user-friendly web application
- [ ] **Streaming analysis**: Process live Twitter data

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Sentiment140 Dataset**: Go et al. (2009) for providing the labeled dataset
- **NLTK Library**: For comprehensive natural language processing tools
- **Scikit-learn**: For machine learning algorithms and utilities
- **Kaggle**: For hosting and providing access to the dataset

---

**Made with ❤️ for the NLP and Machine Learning community**
