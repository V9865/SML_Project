# AG News Text Classification

A comparison of **text representations** and **classical ML classifiers** for 4-class news topic classification on the [AG News](https://huggingface.co/datasets/ag_news) dataset (World, Sports, Business, Sci/Tech), including how much accuracy survives **dimensionality reduction**.

Built for the Statistical Machine Learning course at IIIT Bangalore.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Transformers-FFD21E?logo=huggingface&logoColor=black)

---

## Pipeline

```
AG News (8,000 articles) ─► Clean + tokenize ─► 80/20 split
                                                     │
                 ┌───────────────────┬───────────────┴──────────┐
              Bag-of-Words        TF-IDF (1–2 grams)     DistilBERT embeddings
              (20k features)      (20k features)          (768-dim [CLS])
                 └───────────────────┴──────────┬──────────────┘
                                                ▼
          Logistic Regression · Linear SVM · RBF SVM · KNN (cosine)
                                                │
                    Truncated SVD (sparse) / PCA (dense) at 50, 100, 200 dims
                                                │
                       Accuracy, macro-F1, confusion matrices, plots
```

## What it compares

| Step | Options |
| --- | --- |
| Preprocessing | Lowercasing, noise removal, NLTK tokenization, stopword removal, vocabulary pruning |
| Features | Bag-of-Words, TF-IDF with unigrams + bigrams, DistilBERT `[CLS]` embeddings |
| Classifiers | Logistic Regression, Linear SVM, RBF-kernel SVM, KNN with cosine distance |
| Dimensionality reduction | Truncated SVD on TF-IDF, PCA on embeddings, at 50 / 100 / 200 components |
| Evaluation | Accuracy, macro-F1, per-class F1, confusion matrices |

## Results

Running the pipeline saves these plots to a `plots/` folder:

- `model_comparison_bar.png`: accuracy and macro-F1 for every model
- `feature_comparison.png`: TF-IDF vs. DistilBERT for each classifier
- `dim_reduction_accuracy.png`: accuracy as the number of components changes
- `confusion_matrices.png`: confusion matrix for each model
- `per_class_f1_heatmap.png`: F1 per class and model

<!-- After running, add your best numbers here, e.g.:
| Features | Best model | Accuracy | Macro-F1 |
| --- | --- | --- | --- |
| TF-IDF | Linear SVM | 0.xx | 0.xx |
| DistilBERT | Logistic Regression | 0.xx | 0.xx |
-->

## Project structure

```
code/
  main.py                    runs the full pipeline end to end
  ag_news_preprocessing.py   loading, cleaning, tokenization, train/test split
  ag_news_features.py        Bag-of-Words and TF-IDF features
  ag_news_embeddings.py      DistilBERT / BERT sentence embeddings
  ag_news_classifiers.py     model definitions and training
  ag_news_dim_reduction.py   SVD and PCA experiments
  ag_news_evaluation.py      metrics, reports and plots
```

---

## Getting started

```bash
git clone https://github.com/V9865/SML_Project.git
cd SML_Project

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cd code
python main.py
```

The dataset and the DistilBERT model download automatically on the first run. On a laptop, the RBF SVM on TF-IDF is skipped by default because it is slow; set `skip_rbf_tfidf=False` in `main.py` to include it (a GPU machine such as Kaggle is recommended).
