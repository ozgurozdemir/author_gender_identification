from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import NMF
from sklearn.random_projection import GaussianRandomProjection

from deep_learning_models import *

# Dimensionality reductions
def get_dim_reduction(dim_red, dim_red_param):
    if dim_red == "pca":
        return PCA(**dim_red_param)
    elif dim_red == "lda":
        return LinearDiscriminantAnalysis(**dim_red_param)
    elif dim_red == "nmf":
        return NMF(**dim_red_param)
    elif dim_red == "rp":
        return GaussianRandomProjection(**dim_red_param)
    else:
        return None

# Classifiers
def get_classifier(clf, clf_param):
    if clf == "knn":
        return KNeighborsClassifier(**clf_param)
    elif clf == "nb":
        return BernoulliNB(**clf_param)
    elif clf == "svm":
        return SVC(**clf_param)
    elif clf == "dt":
        return DecisionTreeClassifier(**clf_param)
    elif clf == "rf":
        return RandomForestClassifier(**clf_param)
    elif clf == "sgd":
        return SGDClassifier(**clf_param)
    elif clf == "lr":
        return LogisticRegression(**clf_param)
    elif clf == "boosting":
        return GradientBoostingClassifier(**clf_param)
    elif clf == "bagging":
        return BaggingClassifier(KNeighborsClassifier(**clf_param))
    elif clf == "rocchio":
        return NearestCentroid(**clf_param)
    elif clf == "mlp":
        return MLPClassifier(**clf_param)
    elif clf == "cnn":
        return DeepLearningClassifier("cnn", **clf_param)
    elif clf == "rnn":
        return DeepLearningClassifier("rnn", **clf_param)
    elif clf == "birnn":
        return DeepLearningClassifier("birnn", **clf_param)
    elif clf == "gru":
        return DeepLearningClassifier("gru", **clf_param)
    elif clf == "bigru":
        return DeepLearningClassifier("bigru", **clf_param)
    elif clf == "lstm":
        return DeepLearningClassifier("lstm", **clf_param)
    elif clf == "bilstm":
        return DeepLearningClassifier("bilstm", **clf_param)
    elif clf == "rcnn":
        return DeepLearningClassifier("rcnn", **clf_param)
    elif clf == "hatt":
        return DeepLearningClassifier("hatt", **clf_param)
    
