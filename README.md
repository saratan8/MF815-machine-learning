# MF815-machine-learning

# Mutual Fund Style Classification from Prospectus using NLP and CNN 

This project applies Natural Language Processing (NLP) and deep learning techniques to classify mutual funds into investment strategy categories using the textual summary from their prospectuses.

## Project Overview

We analyze over 500 mutual fund summaries and classify each into one of four investment strategies:

- Balanced Fund (Low Risk)
- Fixed Income Long Only (Low Risk)
- Equity Long Only (Low Risk)
- Long Short Funds (High Risk)

The core idea is to **extract key linguistic features from fund summaries** and use a **Convolutional Neural Network (CNN)** to learn and predict their investment styles.

---

## Data

- `MutualFundSummary/`: Contains ~500 fund summaries (text)
- `MutualFundLabels.csv`: Labels for each fund (converted to 0–3 for classification)

We drop outliers (e.g., the single commodity fund) and split the remaining data 80/20 into train/test sets.

---

## Methodology

### 1. **Text Preprocessing & Embedding**
- Clean text with tokenization, stopword removal, and lemmatization
- Build a vocabulary and train a **skip-gram model** to generate word embeddings (`our_word2vec.txt`)
- Compare custom embeddings with pretrained GloVe vectors (`glove.6B.50d.txt`)

### 2. **Knowledge Base Construction**
- For each class, use **TF-IDF** to identify 10 unique keywords
- Build 4 knowledge bases representing each investment strategy

### 3. **Sentence Selection**
- From each fund summary, extract top 5 sentences for each of the 4 strategies (total of 20 per summary)
- These sentences form the input features for the CNN

### 4. **CNN Model Architecture**
- Two 1D convolutional layers + two dense layers
- Hyperparameter tuning: batch size from 8 to 20 (best = 10)
- Trained model achieved:
  - **In-sample accuracy**: 82.7%
  - **Out-of-sample accuracy**: 49.5%

---

## 📊 Results & Discussion

- Imbalanced data caused model to overfit to dominant classes
- Future improvements may include:
  - Better keyword selection strategies
  - Reducing sentence count per class
  - Model simplification to improve generalization

---
