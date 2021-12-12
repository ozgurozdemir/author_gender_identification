import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from embedding_factory import *

def prepare_dataset_raw(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
        data = data.split('\n')

    comments = []; labels = []
    for sample in data[:-1]:
        sample = sample.split('|')
        comments.append(sample[1])
        labels.append(1 if sample[0] == 'woman' else 0)

    return comments, np.array(labels)


def prepare_dataset_sf(file_path):
    dataset = pd.read_csv(file_path)
    features = dataset.columns[:-1]

    X = dataset[features].values
    y = dataset['class'].values
    y[y=='man'] = 0; y[y=='woman'] = 1

    return normalize(X), y.astype(np.int)


def prepare_dataset_bow(file_path, embedding_params):
    X, y = prepare_dataset_raw(file_path)
    embedding = BoW_Embedding(**embedding_params)

    X = embedding.fit_transform(X)
    return X.toarray(), y


def prepare_dataset_word2vec(file_path, embedding_params):
    X, y = prepare_dataset_raw(file_path)
    embedding = Word2Vec_Embedding(**embedding_params)

    X = embedding.transform_X(X)
    return X.toarray(), y


def prepare_dataset_doc2vec(file_path, embedding_params):
    X, y = prepare_dataset_raw(file_path)
    embedding = Doc2Vec_Embedding(**embedding_params)

    X = embedding.transform_X(X)
    return X.toarray(), y
