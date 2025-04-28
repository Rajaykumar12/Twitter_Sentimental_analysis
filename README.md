# Twitter Sentiment Analysis

A machine learning project that performs sentiment analysis on Twitter data using Logistic Regression. The model classifies tweets as either positive or negative.

## Overview

This project uses the Sentiment140 dataset from Kaggle to train a sentiment analysis model. The model processes tweets and classifies them into two categories:
- 0: Negative sentiment
- 1: Positive sentiment

## Requirements

```python
pip install -r requirements.txt
```

The following packages are required:
- pandas
- numpy
- nltk
- scikit-learn
- matplotlib
- seaborn
- kaggle
- opendatasets

## Dataset

The project uses the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) which contains 1.6 million tweets with their sentiment labels.

## Features

1. **Text Preprocessing**
   - Removal of special characters
   - Converting to lowercase
   - Removing stopwords
   - Stemming using Porter Stemmer

2. **Model Pipeline**
   - TF-IDF Vectorization
   - Train-Test Split (80-20)
   - Logistic Regression Classification

## Model Performance

- Training Accuracy: ~79%
- Testing Accuracy: ~79%

## Usage

1. Clone the repository
2. Install dependencies
3. Download the dataset using Kaggle API
4. Run the Jupyter notebook

## Model Deployment

The trained model is saved using pickle and can be loaded for future predictions:

```python
import pickle

# Load the saved model
loaded_model = pickle.load(open("trained_model.sav", 'rb'))

# Make predictions
prediction = loaded_model.predict(X_new)
```

## Project Structure

- `Sentiment_analysis.ipynb`: Main notebook containing all the code
- `trained_model.sav`: Saved model file
- `requirements.txt`: List of required packages

## Future Improvements

- Implement more advanced preprocessing techniques
- Try different classification algorithms
- Add support for multi-class sentiment analysis
- Improve model accuracy
- Add real-time tweet analysis capability

## License

This project is open source and available under the MIT License.
