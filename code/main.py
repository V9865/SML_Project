from ag_news_preprocessing import load_ag_news, preprocess_corpus, split_data
from ag_news_features import extract_features
from ag_news_embeddings import get_embeddings
from ag_news_classifiers import run_all
from ag_news_dim_reduction import run_dim_reduction
from ag_news_evaluation import run_evaluation


# -------------------------------
# 1. LOAD DATA (HuggingFace)
# -------------------------------
texts, labels = load_ag_news()

# -------------------------------
# 2. PREPROCESS
# -------------------------------
cleaned_texts, token_lists, vocab = preprocess_corpus(texts)

# -------------------------------
# 3. TRAIN-TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = split_data(cleaned_texts, labels)


# -------------------------------
# 4. TF-IDF FEATURES
# -------------------------------
X_bow_tr, X_bow_te, X_tfidf_tr, X_tfidf_te, bow_vec, tfidf_vec = \
    extract_features(X_train, X_test)


# -------------------------------
# 5. BERT EMBEDDINGS
# -------------------------------
embeddings = get_embeddings(X_train, X_test, model_key="distilbert")

X_bert_tr = embeddings["train"]["cls"]
X_bert_te = embeddings["test"]["cls"]


# -------------------------------
# 6. TRAIN MODELS
# -------------------------------
from ag_news_classifiers import run_all
results = run_all(
    X_tfidf_tr, X_tfidf_te,
    X_bert_tr,  X_bert_te,
    y_train, y_test,
    skip_rbf_tfidf=True
) #keep true on laptop false only on kaggle

# -------------------------------
# 7. DIMENSION REDUCTION
# -------------------------------
dim_results = run_dim_reduction(
    X_tfidf_tr, X_tfidf_te,
    X_bert_tr,  X_bert_te,
    y_train, y_test
)


# -------------------------------
# 8. EVALUATION + PLOTS
# -------------------------------
run_evaluation(results, dim_results, y_test)