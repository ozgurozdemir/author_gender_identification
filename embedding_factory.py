from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec, Word2Vec
from sklearn.preprocessing import scale

from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import random
import time

# ==============================================================================
#                                     BoW
# ==============================================================================
class BoW_Embedding():
    def __init__(self, **kwargs):
        self.embedding = TfidfVectorizer(**kwargs)

    def fit_transform(self, X):
        start_time = time.time()
        print(">> Generating BoW embeddings...")

        X = self.embedding.fit_transform(X)
        print(f":: BoW embedding is created in {time.time()-start_time} sec..")
        return X


# ==============================================================================
#                              Word2Vec, Doc2Vec
# ==============================================================================
class Vector_Embeddings():
  def __init__(self, **kwargs):
    self.pretrained_path = kwargs.get('pretrained_path', None)

    # TODO: Loading saved model
    if self.pretrained_path:
      self.model = self.model.load(self.pretrained_path)

  def train(self, X, epochs=10, save_path=None):
    """
      Parameters
      ---------
        X:                (for Word2Vec, str) (for Doc2Vec, TaggedLineSentence) sentences
        epochs (int):     total epoch number to train embedding
        save_path (str):  saving path for trained embedding
    """
    if self.__class__ == Doc2Vec:
      self.model.build_vocab(X.to_array()) # building the vocabulary
      X = X.perm()
    else:
      self.model.build_vocab(X) # building the vocabulary

    start_time = time.time()
    for epoch in range(epochs):
      print(f">> Embedding training >> Epoch {epoch} is started...")
      self.model.train(X, total_examples=self.model.corpus_count, epochs=self.model.epochs)

    print(f">> Embedding training is completed: Time spent {time.time()-start_time} sec...")
    if save_path: self.model.save(save_path)

  def fit_transform(self, X, **kwargs):
    self.train(X, **kwargs)
    return self.transform_X(X)


class Word2Vec_Embedding(Vector_Embeddings):
  def __init__(self, **kwargs):
    # self.size = size
    # self.min_count = min_count
    # self.workers = workers
    # self.window = window
    self.size = kwargs["size"]
    self.min_count = kwargs["min_count"]
    self.workers = kwargs["workers"]
    self.window = kwargs["window"]

    self.model = Word2Vec(size = self.size, min_count = self.min_count, workers = self.workers, window = self.window)
    super(Word2Vec_Embedding, self).__init__(**kwargs)

  def transform_X(self, X):
    # Build word vector for training set by using the average value of all word vectors and scale
    def buildWordVector(text):
      vec = np.zeros(self.size).reshape((1, self.size))
      count = 0.
      for word in text:
          try:
              vec += self.model[word].reshape((1, self.size))
              count += 1.
          except KeyError:
              continue
      if count != 0:
          vec /= count
      return vec

    start_time = time.time()
    print(f">> Word2Vec transforming started...")

    vecs = np.concatenate([buildWordVector(z) for z in X])
    print(f">> Word2Vec transforming ended: Time spent {time.time()-start_time} sec...")
    return scale(vecs)


class Doc2Vec_Embedding(Vector_Embeddings):
  def __init__(self, **kwargs):
    # self.size = size
    # self.min_count = min_count
    # self.workers = workers
    self.size = kwargs["size"]
    self.min_count = kwargs["min_count"]
    self.workers = kwargs["workers"]

    self.model = Doc2Vec(size = self.size, min_count = self.min_count, workers = self.workers)
    super(Doc2Vec_Embedding, self).__init__(**kwargs)

  def transform_X(self, sources):
    sentences = TaggedLineSentence(sources)
    self.model.build_vocab(sentences.to_array())

    for epoch in range(10):
        self.model.train(sentences.sentences_perm(),
                         total_examples=self.model.corpus_count,
                         epochs=self.model.iter)

    data_size=21728
    data_size_train=10000
    data_size_test=1728
    data_size_test_half=864

    train_arrays = np.zeros((data_size, n_dim))
    train_labels = np.zeros(data_size)

    for i in range(data_size_train):
        prefix_train_pos = 'TRAIN_POS_' + str(i)
        prefix_train_neg = 'TRAIN_NEG_' + str(i)
        train_arrays[i] = self.model.docvecs[prefix_train_pos]
        train_arrays[data_size_train + i] = self.model.docvecs[prefix_train_neg]
        train_labels[i] = 1
        train_labels[data_size_train + i] = 0

    #log.info(train_labels)

    test_arrays = np.zeros((data_size_test, n_dim))
    test_labels = np.zeros(data_size_test)

    for i in range(data_size_test_half):
        prefix_test_pos = 'TEST_POS_' + str(i)
        prefix_test_neg = 'TEST_NEG_' + str(i)
        test_arrays[i] = self.model.docvecs[prefix_test_pos]
        test_arrays[data_size_test_half + i] = self.model.docvecs[prefix_test_neg]
        test_labels[i] = 1
        test_labels[data_size_test_half + i] = 0

    X = train_arrays
    y = train_labels


class TaggedLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return(self.sentences)

    def sentences_perm(self):
        shuffled = list(self.sentences)
        random.shuffle(shuffled)
        return(shuffled)


# TODO: Deep Learning embeddings, BERT etc.
class DeepEmbeddings():
  pass
