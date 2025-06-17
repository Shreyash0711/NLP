# Spam Classifier using Multinomial Naive Bayes

This project implements a **Spam Message Classifier** that predicts whether an SMS message is spam or not using **Multinomial Naive Bayes** and **TF-IDF vectorization**.

## Dataset
- **Source**: [SMS Spam Collection Dataset](https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv)
- 2 classes:  
  - **Spam** (1)  
  - **Ham (Not Spam)** (0)


## Model Performance

- **Accuracy**: 97.31%
- **Classification Report**:

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Ham   | 0.97      | 1.00   | 0.98     |
| Spam  | 1.00      | 0.81   | 0.89     |

## Requirements
## pip install numpy pandas scikit-learn nltk joblib

## Notes
- Preprocessing: Lowercasing, punctuation removal, stopword removal, tokenization.
- TF-IDF used for feature extraction.
- Classifier: Multinomial Naive Bayes.
