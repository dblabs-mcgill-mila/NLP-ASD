import os
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.base import TransformerMixin
from sklearn.metrics import f1_score

from gensim.utils import simple_preprocess
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument

import torch
from torch import nn

from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

from helpers.hierarchical_tokenizer import hierarchical_tokenizer
from helpers.hierarchical_tokenizer import process_clean_data
from helpers.batcher import create_batches
from custom_models.sentence_attention_base_pool import SentenceAttentionBERT

from matplotlib import pyplot as plt

# Local path to files
ROOT = "../"
DATA_PATH = os.path.join(ROOT, "data")
REPORT_PATH = os.path.join(DATA_PATH, "reports_txt")

report_files = []
for folder in os.listdir(REPORT_PATH):
    for file in os.listdir(os.path.join(REPORT_PATH, folder)):
        if file.endswith(".txt"):
            report_files.append(os.path.join(REPORT_PATH, folder, file))

meta_frame = pd.read_csv(
    os.path.join(DATA_PATH, "metadata.csv")
)

# clean and process the data
sentence_data = process_clean_data(report_files, meta_frame, sentence_min_length = 30)
sentence_data['labels'] = sentence_data['labels'].astype(int)

# more preprocessing, remove stopwords (necessary for simple baselines)
simple_data = sentence_data.copy()

stop_words = set(stopwords.words('french'))
simple_data["text"] = simple_data["text"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
def lemmatize_text(text):
    st = ""
    for w in w_tokenizer.tokenize(text):
        st = st + lemmatizer.lemmatize(w) + " "
    return st

simple_data["text"] = simple_data.text.apply(lemmatize_text)

class DenseTransformer(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self
    def transform(self, X, y=None, **fit_params):
        return X.toarray()
    
def simple_classifier_cv(text, labels, groups=None, clf=LogisticRegression(), n_splits=5):
    skf = StratifiedGroupKFold(n_splits = n_splits, shuffle = True, random_state = 22)
    avg_acc = []
    avg_f1 = []
    for train_index, test_index in skf.split(text, labels, groups = groups):
        # split into train and test
        X_train, X_test = np.array(text[train_index]), np.array(text[test_index])
        y_train, y_test = np.array(labels[train_index]), np.array(labels[test_index])

        # fit the model
        model = make_pipeline(CountVectorizer(), DenseTransformer(), clf)
        model.fit(X_train, y_train)

        # score the model
        print(model.score(X_test, y_test))
        avg_acc.append(model.score(X_test, y_test))

        # f1
        y_pred = model.predict(X_test)
        avg_f1.append(f1_score(y_test, y_pred))
        
    print(f"=== {np.mean(avg_acc)} ===")

    return avg_acc, avg_f1

results_dict = {}

model_accs, model_f1s = simple_classifier_cv(simple_data.text, simple_data.labels, clf=GaussianNB(), groups = simple_data.pat_id)
results_dict['BOW Naive Bayes'] = (model_accs, model_f1s)

model_accs, model_f1s = simple_classifier_cv(simple_data.text, simple_data.labels, clf=RandomForestClassifier(n_jobs=-1, max_depth=10), groups = simple_data.pat_id, n_splits=5)
results_dict['BOW Random Forest'] = (model_accs, model_f1s)

w2v_data = sentence_data.copy()
w2v_data["text"] = w2v_data["text"].apply(lambda x: simple_preprocess(x))

reports = w2v_data["text"].tolist()
y = np.array(w2v_data["labels"].tolist())

w2v_model = Word2Vec(sentences=reports, vector_size=100, window=5, min_count=5, sg=0, epochs=10) # Tune HPs for full benchmark

def report_to_vector(sentence):
    vector = np.zeros(w2v_model.vector_size)
    for word in sentence:
        if word in w2v_model.wv:
            vector += w2v_model.wv[word]
    return vector / len(sentence)

X = np.array([report_to_vector(report) for report in reports])

skf = StratifiedGroupKFold(n_splits = 5, shuffle = True, random_state = 22)
avg_acc = []
for train_index, test_index in skf.split(w2v_data.text, w2v_data.labels, groups = w2v_data['pat_id']):
    # split into train and test
    X_train, X_test = np.array(X[train_index]), np.array(X[test_index])
    y_train, y_test = np.array(y[train_index]), np.array(y[test_index])

    # fit the model
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # score the model
    print(clf.score(X_test, y_test))
    avg_acc.append(clf.score(X_test, y_test))
print(f"=== {np.mean(avg_acc)} ===")

results_dict['Word2Vec'] = (avg_acc, None)

tagged_data = [TaggedDocument(words=report, tags=[i]) for i, report in enumerate(reports)]
d2v_model = Doc2Vec(tagged_data, vector_size=100, min_count=5, epochs=8) # Tune HPs for full benchmark
d2v_model.build_vocab(tagged_data)

d2v_model.train(tagged_data, total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs)

vectors = [d2v_model.infer_vector(doc) for doc in reports]
X = np.array(vectors)

skf = StratifiedGroupKFold(n_splits = 5, shuffle = True, random_state = 22)
avg_acc = []
avg_f1 = []
for train_index, test_index in skf.split(X, y, groups = w2v_data['pat_id']):
    # split into train and test
    X_train, X_test = np.array(X[train_index]), np.array(X[test_index])
    y_train, y_test = np.array(y[train_index]), np.array(y[test_index])

    # fit the model
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # score the model
    print(clf.score(X_test, y_test))
    avg_acc.append(clf.score(X_test, y_test))

    # f1
    y_pred = clf.predict(X_test)
    avg_f1.append(f1_score(y_test, y_pred))
print(f"=== {np.mean(avg_acc)} ===")

results_dict['Doc2Vec'] = (avg_acc, avg_f1)