# rcv1 multilabel nn
Train a simple neural network using RCV1 data and compare it with a logistic regression baseline.

# RCV1 Multi-Label Text Classification  
*(Neural Network vs Logistic Regression Baseline)*

This project trains a **feed-forward neural network** on the **Reuters RCV1** dataset and compares it with a **One-vs-Rest Logistic Regression** baseline for **multi-label topic classification**.

---

## 🔍 Problem & Business Context

**Business-style problem**

Given a stream of news articles, we want to automatically assign relevant topic tags  
(for example: *Economics*, *Markets*, *Corporate*, *Trade*, etc.).

This can be used for:

- News search and personalised recommendations  
- Tagging content for analytics dashboards  
- Routing articles to the correct editorial / research teams  
- Building alert systems for specific topics (e.g. “Oil & Gas”, “Interest Rates”)

**Why a neural network?**

- The RCV1 dataset is **high-dimensional** (47k+ TF-IDF features) and **multi-label** (103 topics).
- Logistic Regression is a strong **linear baseline**, but cannot easily capture complex, non-linear relationships between words and topics.
- A neural network can model these nonlinear patterns and shared representations across topics, often improving recall and F1.

We use **Logistic Regression** as a **baseline** and show how a **neural network** improves performance.

---

## 📊 Models

### 1. Logistic Regression — One-vs-Rest baseline

- One binary classifier per label (`topic` vs `not topic`)
- Good **baseline**: fast, interpretable, simple
- Handles multi-label output via `OneVsRestClassifier(LogisticRegression)`

Example metrics on the sampled data (fill with your actual results):

- PR-AUC (micro): `0.1376`
- F1 (micro): `0.3064`
- F1 (macro): `0.2892`

---

### 2. Neural Network — Keras / TensorFlow

Architecture used in this project:

- **Input**: 47,236 TF-IDF features (sparse → dense array)
- `Dense(512, activation="relu")`
- `Dropout(0.3)`
- `Dense(128, activation="relu")`
- `Dropout(0.3)`
- `Dense(103, activation="sigmoid")`  ← one neuron per label (multi-label)

Training details:

- Loss: `binary_crossentropy` (suitable for multi-label)
- Optimizer: `adam`
- Metric: `tf.keras.metrics.AUC(curve="PR", multi_label=True, name="pr_auc")`
- Dataset split: sample 10,000 documents → 80% train / 20% test

Example metrics :

- PR-AUC (micro): `0.58`
- F1 (micro): `0.82`
- F1 (macro): `0.45`

The neural network significantly improves over the logistic baseline, showing the benefit of learning non-linear interactions between features.

---

## 💻 Run in Google Colab

The notebooks in this repo can be opened directly in Colab:

### Neural Network Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KrysD09/rcv1-multilabel-nn/blob/main/notebooks/nn_training_baseline.ipynb)

### Logistic Regression Baseline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KrysD09/rcv1-multilabel-nn/blob/main/notebooks/nn_training_alt.ipynb)


---

## 📁 Project Structure

```text
rcv1-multilabel-nn/
│
├─ notebooks/
│   ├─ rcv1_neural_network.ipynb        # NN in Colab
│   └─ rcv1_logistic_baseline.ipynb     # LR baseline in Colab
│
├─ requirements.txt                     # Python dependencies for this project
├─ README.md                            # This file
└─ .gitignore                           # (Python template – auto from GitHub)

