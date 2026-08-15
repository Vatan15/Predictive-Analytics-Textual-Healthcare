# Predictive Analytics on Textual & Healthcare Datasets

Two independent supervised learning problems tackled using the CRISP-DM
methodology: multi-class sentiment classification on German medical
practitioner reviews, and imbalanced-class stroke risk prediction on
patient health data.

## Tech Stack
- Python, Pandas, NumPy, Scikit-Learn
- TF-IDF vectorization, Logistic Regression, Linear SVM
- Random Forest, permutation feature importance

## Part 1: Sentiment Classification (German Doctor Reviews)
- Built a multi-class (0-5 rating) sentiment classifier using TF-IDF +
  Logistic Regression / Linear SVM
- Logistic Regression reached 85.9% accuracy; macro-F1 of 0.40 highlighted
  the impact of class imbalance on minority sentiment classes
- Extracted top positive/negative predictive words for interpretability

## Part 2: Stroke Risk Prediction
- Compared Logistic Regression (interpretable baseline) vs. Random Forest
  on a highly imbalanced clinical dataset
- Random Forest achieved AUC 0.962 and F1 0.563 (vs. Logistic Regression's
  F1 of 0.0), dramatically improving minority-class (stroke-positive) recall
- Used permutation importance to identify age, glucose level, and heart
  disease as the top clinical risk factors

## Files
- `Notebooks/` — full analysis code
- `report.pdf` — full written report
- `Datasets/` — datasets used

## Course Context
Built for the Data Mining and Machine Learning module, MSc Data Analytics,
National College of Ireland.

**Note:** `2021_german_doctor_reviews.csv` (160MB) is excluded due to
GitHub's 100MB file size limit. The stroke prediction dataset is included.