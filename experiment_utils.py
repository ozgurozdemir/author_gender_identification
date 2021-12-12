import pandas as pd
import numpy as np
from datetime import datetime
import time

from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from dataset_utils import *


def run_experiments(feature_mode, learning_mode, dataset_path,
                    classifiers, dim_reductions,
                    save_dir, verbose=False, embedding_params=None):
    # Dataset preparation
    if feature_mode == "sf":
        X, y = prepare_dataset_sf(dataset_path)
    elif feature_mode == "raw":
        X, y = prepare_dataset_raw(dataset_path)

    elif feature_mode == "bow":
        X, y = prepare_dataset_bow(dataset_path, embedding_params)
    elif feature_mode == "wor2vec":
        X, y = prepare_dataset_word2vec(dataset_path, embedding_params)
    elif feature_mode == "doc2vec":
        X, y = prepare_dataset_doc2vec(dataset_path, embedding_params)

    # Initialize table for results
    results = init_results_table(classifiers, dim_reductions)
    metrics = ['accuracy', 'recall', 'precision', 'f1']
    current_time = f"({datetime.now().day}-{datetime.now().month})_{datetime.now().hour}_{datetime.now().minute}"
    save_path = f"{save_dir}/exp_{feature_mode}_{learning_mode}_{current_time}.csv"

    # Run experiments
    for clf in classifiers:

        # machine learning training & testing
        if learning_mode == "ml":
            for dim in dim_reductions:
                pipeline = Pipeline([('dim_reduction', dim_reductions[dim]),
                                     ('clf', classifiers[clf])],
                                    verbose=verbose)
                scores = cross_validate(pipeline, X, y, scoring=metrics, cv=10, verbose=verbose)

                for m in metrics:
                    results[f"{clf}+{dim}"][m] = scores[f'test_{m}'].mean()

        # deep learning training & testing
        elif learning_mode == "deep":
            scores = classifiers[clf].cross_validate(X, y, metrics, cv=10)

            for m in metrics:
                results[f"{clf}"][m] = np.mean(scores[f'test_{m}'])

        # save test results
        pd.DataFrame(results).to_csv(save_path)

    # Results of the experiments
    if verbose:
        print("=" * 30 + "\nResult:\n" + f"{pd.DataFrame(results)}")


def init_results_table(classifiers, dim_reductions):
    results = {}

    if len(dim_reductions) > 0:
      for dim in dim_reductions:
          for clf in classifiers:
              results[f"{clf}+{dim}"] = {"accuracy": -1.0, "recall": -1.0,
                                         "precision": -1.0, "f1": -1.0}
    else:
      for clf in classifiers:
          results[f"{clf}"] = {"accuracy": -1.0, "recall": -1.0,
                               "precision": -1.0, "f1": -1.0}
    return results
