from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

import tensorflow.keras.backend as K
import tensorflow as tf

import numpy as np
import time


class DeepLearningClassifier():
    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs


    def tokenize(self, X, y):
        start_time = time.time()
        print(">> Tokenization is started...")
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=15000)

        # train tokenizer
        self.tokenizer.fit_on_texts(X)
        self.vocab_size = len(self.tokenizer.word_index) + 1

        # pad/trim sequences
        X = self.tokenizer.texts_to_sequences(X)
        X = tf.keras.preprocessing.sequence.pad_sequences(X, padding='post',
                                                          maxlen=self.kwargs["maxlen"])
        y = tf.keras.utils.to_categorical(y)
        print(f":: Tokenization is ended in {time.time()-start_time} sec...")

        return X, y

    def cross_validate(self, X, y, metric, cv=10):
        scores = {f"test_{m}": [] for m in metric}

        # tokenize the sentences
        if self.model == "hatt":
            X, y = prepare_dataset_sent_for_hatt(X, y)
        else:
            X, y = self.tokenize(X, y)

        # extract folds
        kf = KFold(n_splits=cv)
        for i, (tr_idx, ts_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[tr_idx], X[ts_idx]
            y_train, y_test = y[tr_idx], y[ts_idx]

            # re-create the model
            classifier = self.create_model()

            # train the model
            classifier.fit(X_train, y_train,
                           epochs=self.kwargs["epochs"],
                           batch_size=self.kwargs["batch_size"], verbose=False)

            # test the model
            acc, f1, prec, recall = self.evaluate_model(classifier, X_test, y_test, verbose=False)

            scores["test_accuracy"].append(acc)
            scores["test_recall"].append(recall)
            scores["test_precision"].append(prec)
            scores["test_f1"].append(f1)

            if self.kwargs["verbose"]:
                print(f'>> {i+1}-th fold : f-1 => {f1}')

        return scores


    def create_model(self):
        if self.model == "cnn":
            return create_CNN_model(self.vocab_size, self.kwargs)
        elif self.model == "rnn":
            return create_RNN_model(**self.kwargs)


    def evaluate_model(self, classifier, x_test, y_test, verbose=False):
        # predictions
        predict = classifier.predict(x_test, verbose=0)

        predict = [np.argmax(i) for i in predict]
        y_test  = [np.argmax(i) for i in y_test]

        cnfs = confusion_matrix(y_test, predict)
        tn = cnfs[0,0]; tp = cnfs[1,1];
        fn = cnfs[1,0]; fp = cnfs[0,1];

        # calculate acc, recall, prec, and f1
        acc = (tp + tn) / (tp + tn + fp + fn)
        prec = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * prec * recall / (prec + recall)

        if verbose:
            print('-'*50, '\nConfussion Matrix:')
            print(confusion_matrix(y_test, predict), '\n', '-'*50)
            print(classification_report(y_test, predict))

        return acc, f1, prec, recall

# ==============================================================================
#                                CNN model
# ==============================================================================
def create_CNN_model(vocab_size, kwargs):
    embedding_dim = kwargs["embedding_dim"]
    maxlen        = kwargs["maxlen"]
    dropout       = kwargs["dropout"]
    fcn_dim       = kwargs["fcn_dim"]

    # embedding
    inputs = tf.keras.Input(shape=(maxlen,), dtype='float32')
    embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                          output_dim=embedding_dim,
                                          input_length=maxlen)
    inp = embedding(inputs)

    # CNN
    x = tf.keras.layers.Conv1D(embedding_dim, 4, activation='relu',padding='same')(inp)
    x = tf.keras.layers.MaxPooling1D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Flatten()(x)

    y = tf.keras.layers.Conv1D(embedding_dim, 3, activation='relu',padding='same')(inp)
    y = tf.keras.layers.MaxPooling1D()(y)
    y = tf.keras.layers.Dropout(dropout)(y)
    y = tf.keras.layers.Flatten()(y)

    z = tf.keras.layers.Conv1D(embedding_dim, 2, activation='relu',padding='same')(inp)
    z = tf.keras.layers.MaxPooling1D()(z)
    z = tf.keras.layers.Dropout(dropout)(z)
    z = tf.keras.layers.Flatten()(z)

    x = tf.keras.layers.Concatenate()([x, y, z])
    x = tf.keras.layers.Dropout(dropout)(x)

    # FCN
    for d in fcn_dim:
        x = tf.keras.layers.Dense(d, activation='relu')(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)

    # compiling model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    return model


# ==============================================================================
#                                RNN models
# ==============================================================================
def create_RNN_model(vocab_size, kwargs):
    embedding_dim = kwargs["embedding_dim"]
    maxlen        = kwargs["maxlen"]
    dropout       = kwargs["dropout"]
    fcn_dim       = kwargs["fcn_dim"]

    # Embedding
    inputs = tf.keras.Input(shape=(maxlen,), dtype='float32')
    embd = tf.keras.layers.Embedding(input_dim=vocab_size,
                                     output_dim=embedding_dim,
                                     input_length=maxlen)(inputs)

    # RNN
    x = tf.keras.layers.SimpleRNN(embedding_dim, return_sequences=True)(embd)
    x = tf.keras.layers.SimpleRNN(embedding_dimembedding_dim//2)(x)

    # FCN
    for d in fcn_dim:
        x = tf.keras.layers.Dense(d, activation='relu')(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)

    model = tf.keras.models.Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def create_BiRNN_model(vocab_size, kwargs):
    embedding_dim = kwargs["embedding_dim"]
    maxlen        = kwargs["maxlen"]
    dropout       = kwargs["dropout"]
    fcn_dim       = kwargs["fcn_dim"]

    # Embedding
    inputs = tf.keras.Input(shape=(maxlen,), dtype='float32')
    embd = tf.keras.layers.Embedding(input_dim=vocab_size,
                                     output_dim=embedding_dim,
                                     input_length=maxlen)(inputs)

    # RNN
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.SimpleRNN(embedding_dim, return_sequences=True))(embd)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.SimpleRNN(embedding_dimembedding_dim//2))(x)

    # FCN
    for d in fcn_dim:
        x = tf.keras.layers.Dense(d, activation='relu')(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)

    model = tf.keras.models.Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def create_LSTM_model(vocab_size, kwargs):
    embedding_dim = kwargs["embedding_dim"]
    maxlen        = kwargs["maxlen"]
    dropout       = kwargs["dropout"]
    fcn_dim       = kwargs["fcn_dim"]

    # Embedding
    inputs = tf.keras.Input(shape=(maxlen,), dtype='float32')
    embd = tf.keras.layers.Embedding(input_dim=vocab_size,
                                     output_dim=embedding_dim,
                                     input_length=maxlen)(inputs)

    # RNN
    x = tf.keras.layers.LSTM(embedding_dim, return_sequences=True)(embd)
    x = tf.keras.layers.LSTM(embedding_dimembedding_dim//2)(x)

    # FCN
    for d in fcn_dim:
        x = tf.keras.layers.Dense(d, activation='relu')(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)

    model = tf.keras.models.Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def create_BiLSTM_model(vocab_size, kwargs):
    embedding_dim = kwargs["embedding_dim"]
    maxlen        = kwargs["maxlen"]
    dropout       = kwargs["dropout"]
    fcn_dim       = kwargs["fcn_dim"]

    # Embedding
    inputs = tf.keras.Input(shape=(maxlen,), dtype='float32')
    embd = tf.keras.layers.Embedding(input_dim=vocab_size,
                                     output_dim=embedding_dim,
                                     input_length=maxlen)(inputs)

    # RNN
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(embedding_dim, return_sequences=True))(embd)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(embedding_dimembedding_dim//2))(x)

    # FCN
    for d in fcn_dim:
        x = tf.keras.layers.Dense(d, activation='relu')(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)

    model = tf.keras.models.Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def create_GRU_model(vocab_size, kwargs):
    embedding_dim = kwargs["embedding_dim"]
    maxlen        = kwargs["maxlen"]
    dropout       = kwargs["dropout"]
    fcn_dim       = kwargs["fcn_dim"]

    # Embedding
    inputs = tf.keras.Input(shape=(maxlen,), dtype='float32')
    embd = tf.keras.layers.Embedding(input_dim=vocab_size,
                                     output_dim=embedding_dim,
                                     input_length=maxlen)(inputs)

    # RNN
    x = tf.keras.layers.GRU(embedding_dim, return_sequences=True)(embd)
    x = tf.keras.layers.GRU(embedding_dimembedding_dim//2)(x)

    # FCN
    for d in fcn_dim:
        x = tf.keras.layers.Dense(d, activation='relu')(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)

    model = tf.keras.models.Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def create_BiGRU_model(vocab_size, kwargs):
    embedding_dim = kwargs["embedding_dim"]
    maxlen        = kwargs["maxlen"]
    dropout       = kwargs["dropout"]
    fcn_dim       = kwargs["fcn_dim"]

    # Embedding
    inputs = tf.keras.Input(shape=(maxlen,), dtype='float32')
    embd = tf.keras.layers.Embedding(input_dim=vocab_size,
                                     output_dim=embedding_dim,
                                     input_length=maxlen)(inputs)

    # RNN
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(embedding_dim, return_sequences=True))(embd)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(embedding_dimembedding_dim//2))(x)

    # FCN
    for d in fcn_dim:
        x = tf.keras.layers.Dense(d, activation='relu')(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)

    model = tf.keras.models.Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


# ==============================================================================
#                                R-CNN models
# ==============================================================================
def create_RCNN_model(vocab_size, kwargs):
    embedding_dim = kwargs["embedding_dim"]
    maxlen        = kwargs["maxlen"]
    dropout       = kwargs["dropout"]
    fcn_dim       = kwargs["fcn_dim"]

    # Embedding
    inputs = tf.keras.Input(shape=(maxlen,), dtype='float32')
    embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                     output_dim=embedding_dim,
                                     input_length=maxlen)(inputs)

    x = embedding(inputs)

    # CNN
    x = tf.keras.layers.Conv1D(embedding_dim*2, 7, activation='relu',padding='same')(x)
    x = tf.keras.layers.MaxPooling1D()(x)
    x = tf.keras.layers.Conv1D(embedding_dim*4, 5, activation='relu',padding='same')(x)
    x = tf.keras.layers.MaxPooling1D()(x)
    x = tf.keras.layers.Conv1D(embedding_dim*8, 3, activation='relu',padding='same')(x)
    x = tf.keras.layers.MaxPooling1D()(x)

    # RNN
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(embedding_dim*2, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(embedding_dim, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(embedding_dim/2))(x)

    # FCN
    for d in fcn_dim:
        x = tf.keras.layers.Dense(d, activation='relu')(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)

    model = tf.keras.models.Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


# ==============================================================================
#                           Hierarchical Attention models
# souce: https://github.com/richliao/textClassifier/blob/master/textClassifierHATT.py
# ==============================================================================
def create_HATT_model(vocab_size, kwargs):
    embedding_dim = kwargs["embedding_dim"]
    maxlen        = kwargs["maxlen"]
    dropout       = kwargs["dropout"]
    fcn_dim       = kwargs["fcn_dim"]

    inputs = tf.keras.Input(shape=(maxlen,), dtype='float32')
    embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                        output_dim=embedding_dim,
                                        input_length=maxlen)

    x = embedding(inputs)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(embedding_dim*2, return_sequences=True))(x)
    x = AttLayer(embedding_dim*2)(x)

    for d in fcn_dim:
        x = tf.keras.layers.Dense(d, activation='relu')(x)
    outputs = tf.keras.layers.Dense(64, activation='relu')(x)

    wordEncoder = tf.keras.Model(inputs=inputs, outputs=outputs)

    # =============================================================================

    inputs = tf.keras.Input(shape=(10,maxlen), dtype='float32')
    x = tf.keras.layers.TimeDistributed(wordEncoder)(inputs)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(embedding_dim*4, return_sequences=True))(x)

    x = AttLayer(embedding_dim*2)(x)
    for d in fcn_dim:
        x = tf.keras.layers.Dense(d, activation='relu')(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    return model


class AttLayer(tf.keras.layers.Layer):
    def __init__(self, attention_dim):
        self.init = tf.keras.initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)), name='W')
        self.b = K.variable(self.init((self.attention_dim, )), name='b')
        self.u = K.variable(self.init((self.attention_dim, 1)), name='u')
        # self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def prepare_dataset_sent_for_hatt(X, y):
    sentX = []; sentY = []

    # obtain sentences
    for i, x in enumerate(X):
        for sent in x.split('.'):
            words = sent.split()
            sentX.append([w for w in words if len(w) > 0])
            sentY.append(y[i])

    # train tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=15000)
    tokenizer.fit_on_texts(sentX)

    # separate man and woman for labelling
    dt = list(zip(sentX, sentY))
    man   = [i[0] for i in dt if i[1] == 0][:618880]
    woman = [i[0] for i in dt if i[1] == 1][:618880]

    # prepare sentence vectors
    X_sent = []; y_sent = []
    for i in range(1, 61800):
          X_sent.append(tokenizer.texts_to_sequences(man[i*10:(i+1)*10]))
          y_sent.append([0])

          X_sent.append(tokenizer.texts_to_sequences(woman[i*10:(i+1)*10]))
          y_sent.append([1])

    X_sent = X_sent[1:-1]; y_sent= y_sent[1:-1]
    X_sent = np.array(X_sent)
    y_sent =  tf.keras.utils.to_categorical(y_sent)

    # pad sentences
    X = np.zeros((123596, 10, 40), dtype='int16')
    for i, sen in tqdm(enumerate(X_sent[:-1])):
        X[i,:] = pad_sequences(sen, padding='post', maxlen=40)

    return X, y_sent
