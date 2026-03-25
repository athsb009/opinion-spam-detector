# Opinion Spam Detection Using Ensemble Learning

Detecting fake online reviews using classical ML and ensemble learning techniques. This project benchmarks 8 classifiers — from Logistic Regression to XGBoost — and evaluates whether ensemble methods (bagging, AdaBoost) outperform their conventional counterparts on the Amazon Verified Purchase dataset.

## The Problem

Fake reviews manipulate consumer decisions and damage business reputations. Detecting them is a binary classification problem complicated by the inherent complexity of natural language — sarcasm, context, and latent features make rule-based approaches ineffective. This project takes a linguistics-based approach, analyzing review text rather than reviewer behavior.

## Results

- Benchmarked 8 ML classifiers with and without ensemble techniques
- Hyperparameter optimization via 100-iteration randomized search with 10-fold cross-validation
- Ensemble methods (bagging, AdaBoost) consistently outperformed baseline classifiers
- Evaluated using accuracy, precision, recall, and F1 score

## Tech Stack

- **Language:** Python
- **ML Libraries:** Scikit-learn, XGBoost
- **NLP:** TF-IDF, Count Vectorization
- **Data:** Amazon Verified Purchase dataset (Kaggle)

## Models Benchmarked

| Model | Ensemble Variants |
|-------|------------------|
| Logistic Regression | Bagging, AdaBoost |
| Decision Tree | Bagging, AdaBoost |
| Random Forest | Bagging, AdaBoost |
| K-Nearest Neighbors | Bagging |
| Multinomial Naive Bayes | Bagging, AdaBoost |
| Support Vector Machine | Bagging, AdaBoost |
| XGBoost (Gradient Boosting) | Baseline only |
| Multilayer Perceptron | Bagging |

> Note: KNN and MLP were omitted from AdaBoost as they do not support `sample_weight`. XGBoost is already an ensemble boosting technique.

## Pipeline

```
Raw Data (32 columns)
        ↓
Data Cleaning (keep review_text + verified_purchase)
        ↓
Preprocessing (remove stopwords, special chars, punctuation)
        ↓
Feature Extraction (TF-IDF + Count Vectorization)
        ↓
Hyperparameter Optimization (100-iteration randomized search, 10-fold CV)
        ↓
Model Training (8 classifiers × baseline + ensemble variants)
        ↓
Evaluation (accuracy, precision, recall, F1)
```

## Hyperparameter Search Space

Each classifier was optimized independently via randomized search:

- **Logistic Regression** — C ∈ [0.1, 1000), solver ∈ {newton-cg, lbfgs, liblinear, sag, saga}
- **Decision Tree** — max_depth ∈ [3, 5), splitter ∈ {best, random}, criterion ∈ {gini, entropy}
- **Random Forest** — n_estimators ∈ [100, 1000), max_depth ∈ [3, 5)
- **KNN** — n_neighbors ∈ [1, 21), weights ∈ {uniform, distance}
- **Naive Bayes** — alpha ∈ [0.1, 2.0)
- **SVM** — kernel ∈ {RBF, linear, polynomial, sigmoid}, C ∈ [0.1, 1000)
- **XGBoost** — n_estimators ∈ [100, 1000), learning_rate ∈ [0.01, 1.0)
- **MLP** — hidden_layer_sizes sampled per layer, activation ∈ {ReLU, tanh, sigmoid, identity}

## Setup

```bash
# Clone the repo
git clone https://github.com/athsb009/opinion-spam-detection
cd opinion-spam-detection

# Install dependencies
pip install -r requirements.txt

# Download dataset
# https://www.kaggle.com/datasets/akudnaver/amazon-reviews-dataset

# Run preprocessing
python preprocess.py

# Train and evaluate all models
python train.py
```

## Dataset

Amazon Verified Purchase dataset from Kaggle — contains customer reviews across Amazon branded products. Only `review_text` and `verified_purchase` columns are used after initial EDA.

[Dataset Link](https://www.kaggle.com/datasets/akudnaver/amazon-reviews-dataset)

## Key Findings

- Ensemble methods consistently improved classification accuracy over baseline models
- TF-IDF outperformed Count Vectorization for most classifiers
- XGBoost achieved competitive performance as a standalone boosting method
- Hyperparameter optimization via randomized search improved accuracy across all classifiers

## Author

**Atharva Bibave**
- [LinkedIn](https://linkedin.com/in/atharvabibave)
- [GitHub](https://github.com/athsb009)
