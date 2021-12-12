# A Comprehensive Study of Learning Approaches for Author Gender Identification

## Abstract
_In  recent  years,  author  gender  identification  is  an  important  yet  challenging  task  in  the  fields  of information  retrieval  and  computational  linguistics.  In  this  paper,  different  learning  approaches  are presented  to  address  the  problem  of  author  gender  identification  for  Turkish  articles.  First,  several classification algorithms are applied to the list of representations based on different paradigms: fixed-length vector representations such as Stylometric Features (SF), Bag-of-Words  (BoW)  and  distributed word/document  embeddings  such  as  Word2vec,  fastText  and  Doc2vec.  Secondly,  deep  learning architectures, Convolution Neural Network (CNN), Recurrent Neural Network (RNN), special kinds of RNN such as Long-Short Term Memory (LSTM) and Gated Recurrent Unit (GRU), C-RNN, Bidirectional LSTM  (bi-LSTM),  Bidirectional  GRU  (bi-GRU),  Hierarchical  Attention  Networks  and  Multi-head Attention  (MHA)  are  designated  and  their  comparable  performances  are  evaluated.  We  conducted  a variety of experiments and achieved outstandingempirical results. To conclude, ML algorithms with BoW have  promising  results.  fast-Text  is  also  probably  suitable  between  embedding  models.  This comprehensive study contributes to literature utilizing different learning approaches based on several ways of representations. It is also first attempt to identify author gender applying SFon Turkish language._

# Datasets
The dataset is collected from Turkish news articles. Summary of the dataset is given below
below:

|          | Male | Female |
|----------|:-----------:|:-----------:|
| # of authors |    145    |    145   |
| # of articles   |   10864   |    10864   |
| # of words / article   |   575.91   |    624.66   |
| # of words / sentences   |   16.57   |    14.52   |


# Experimental Results
The experiments were conducted by 10-fold cross-validation preserving 20% of the dataset for each iteration. The best results are as given below

| Feature  | Classifier | Dimensionality Reduction |  F1-score  |
|----------|:-----------:|:-----------:|:------:|
| SF |    RF   |    -   | 0.80 |
|   BoW-tfidf   |    k-NN   |    -   | 0.87 |
|   Word2Vec   |    SVM   |    PCA   | 0.77 |
|   Doc2Vec   |    SVM   |    PCA   | 0.84 |
|   Word Embedding   |    CNN   |    -   | 0.83 |
|   fastText   |    fastText<sup>1</sup>   |    -   | 0.87 |

<sup>1</sup>_Source: https://github.com/facebookresearch/fastText/_

# Implementation
* [```dataset_utils.py```](dataset_utils.py): Provides functions for dataset read & preparation.
* [```embedding_factory.py```](embedding_factory.py): Provides interface for BoW, Word2Vec and Doc2Vec embeddings.
* [```classifier_factory.py```](classifier_factory.py): Provides interface for classifiers.
* [```deep_learning_models.py```](deep_learning_models.py): Provides implementation of deep learning models: RNN, CNN, C-RNN, Multi-head Attention and Hierarchical Attention<sup>2</sup>
* [```params.json```](params.json): Configurations of the experiments.
* [```experiment_utils.py```](experiment_utils.py): Provides function for running multiple experiment which configurations are given in [```params.json```](params.json) and saving results.
* [```run_experiments.py```](run_experiments.py): Provides codes for parsing the configurations and running experiments.

<sup>2</sup>_Implementation source: https://github.com/richliao/textClassifier_

# Usage
For the experiment configurations, modify [```params.json```](params.json). Details of the parameters are given below

|  Parameter | Description |
|----------|:-----------:|
| FEATURE_MODE |    raw/sf/bow/word2vec/doc2vec    |
| LEARNING_MODE   |   ml/deep   |
| DATASET_PATH   |   path to dataset   |
| SAVE_DIR   |   path to directory for saving experiment results  |
| EMBEDDINGS   |   embedding parameters. Refer to [BoW](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html), [Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html) and [Doc2Vec](https://radimrehurek.com/gensim/models/doc2vec.html) |
| DIM_REDUCTIONS  |  dimensionality reduction algorithms. Use ```"none": {}``` for running experiments without dimensionality reduction.  |
| CLASSIFIERS   |   classifier algorithms. Refer to  [```classifier_factory.py```](classifier_factory.py) for the list of the available classifiers. |
| VERBOSE   |  feed back information while running the experiments  |

Note that the Deep Learning experiments must run on ```FEATURE_MODE="raw"```.

Once the configuration parameters are fixed, then run experiments:

``` $ python run_experiments.py```

# Citation
_To be shared once the artilce is published_
