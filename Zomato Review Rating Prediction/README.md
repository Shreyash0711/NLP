# NLP
# Zomato Review Rating Prediction

This project predicts restaurant review ratings (1 to 5) based on the text of customer reviews using NLP and Machine Learning (Logistic Regression with TF-IDF).

## Dataset
- **File:** Zomato_reviews.csv
- **Shape:** (27762, 2)
- **Columns:**
  - `rating`: Original rating given by customer
  - `review_text`: Text review given by customer

## Model Performance
- **Accuracy:** 81%
- **Classification Report:** Available in training output.

## Steps
1. Data Cleaning and Preprocessing (stopword removal, punctuation removal, tokenization)
2. TF-IDF Vectorization
3. Logistic Regression Model (multi-class)
4. Model Saving using Joblib
5. Simple CLI-based Interface for rating prediction


#Model Evaluation Summary
Overall Accuracy: 81%

-**Best Predicted Ratings**: 1, 4, 5 (High precision & recall)

-**Weakest Performance:** Rating 2 (Low recall — model misses some true 2’s)

-**F1-Score Range:** 0.72 – 0.88 across classes

-**Macro F1-Score:** 0.80 (balanced across classes)

-**Weighted F1-Score:** 0.81 (accounts for class distribution)

✔️ Model is reliable for general use
⚠️ Needs improvement for detecting Rating 2 & 3 more accurately.
