"""
Churn Benchmark Pipeline
This file benchmarks and saves the results of all four llm models on the churn datasets.
It uses default hyperparameters and does not perform any hyperparameter optimization since
we do a simple k fold cross validation here and do no test set holdout.
Otherwise, the holdout would result in only one test score per dataset and model.
Hence, a Nested Cross Val would be needed.

Author: Nico Hambauer, Mathias Kraus
Created:  21st of November 2022
No license
"""
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Dataset import Dataset


class FiveFoldBenchmark:
    """
    The FiveFoldBenchmark class is used to run the benchmark using a 5-fold cross validation on the respective churn dataset.
    Initialize the class with the respective model and dataset id.
    """

    def __init__(self, model):
        self.model = model
        """Callable model class"""
        self.auroc_scores = []
        """The scores array holds the ROC AUC for the model on the dataset for each fold."""
        self.dataset_id = None
        """ The dataset id is used to load the dataset from the Dataset.py file. In the
        range of 1 to 14. """
        self.X = None
        """The X array holds the features of the dataset."""
        self.y = None
        """The y array holds the labels of the dataset."""
        self.numerical_cols = None
        """The numerical_cols array holds the names of the numerical columns of the dataset."""

    def set_data(self, X, y, numerical_cols):
        self.X = X
        self.y = y
        self.numerical_cols = numerical_cols

    def _load_data(self, dataset_id):
        self.dataset_id = dataset_id
        dataset = Dataset(dataset_id=self.dataset_id)
        self.X, self.y = dataset.X, dataset.y
        self.numerical_cols = dataset.numerical_cols

    def run_model_on_dataset(self, dataset_id=None):
        if dataset_id is not None:
            self._load_data(dataset_id)
        elif self.X is None or self.y is None:
            raise ValueError("Either set the dataset id or set the data manually.")

        rus = RandomUnderSampler(random_state=42)
        X, y = rus.fit_resample(self.X, self.y)
        # num cols should be std scaled already, but we do it again which will have no effect if so
        num_pipe = Pipeline([('scaler', StandardScaler())])
        transformers = [
            ('num', num_pipe, self.numerical_cols)
        ]

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for i, (train_index, test_index) in enumerate(cv.split(X, y)):
            print('-' * 5, 'Model', '-' * 3, 'Dataset', dataset_id, '-' * 3, f'Fold {i + 1}', '-' * 5)
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            ct = ColumnTransformer(transformers=transformers)
            ct.fit(X_train)
            X_train = ct.transform(X_train)
            X_test = ct.transform(X_test)
            print(f"Count statistics on y train: {np.unique(y_train, return_counts=True)}")
            self.model.fit(pd.DataFrame(X_train), y_train)
            #m.plot_learning()
            cl_report = pd.DataFrame(classification_report(y_test, self.model.predict(X_test), output_dict=True)).T
            roc_auc = roc_auc_score(y_test, self.model.predict_proba(X_test)[:,1])
            print(f"Roc Auc: {roc_auc:.4f}")
            self.auroc_scores.append(roc_auc)

        return self.auroc_scores


if __name__ == '__main__':
    model_dict = {
        'IGANN': [
            (f'', '', IGANN(verbose=0))],
    }

    results_fd = open('results.txt', 'a')
    best_results_fd = open('best_results.txt', 'a')

    for base_model, v in model_dict.items():
        results_fd.write(f'd_id;model;{v[0][1]};mean_auroc;std_auroc' + '\n')
        best_results_fd.write(f'd_id;model;{v[0][1]};mean_auroc;std_auroc' + '\n')
        for dataset_id in range(1, 15):
            best_auroc_mean = 0
            best_auroc_std = 0
            best_para = None
            for para_str, param_names, model in v:
                benchmark = FiveFoldBenchmark(model=model)
                folds_auroc = benchmark.run_model_on_dataset(dataset_id=dataset_id)
                results_fd.write(
                    str(dataset_id) + ';' + base_model + ';' + para_str + ';' + f'{np.mean(folds_auroc):.4f}' + ';'
                    + f'{np.std(folds_auroc):.4f}' + '\n')

                if np.mean(folds_auroc) > best_auroc_mean:
                    best_auroc_mean = np.mean(folds_auroc)
                    best_auroc_std = np.std(folds_auroc)
                    best_para = para_str

            best_results_fd.write(
                str(dataset_id) + ';' + base_model + ';' + best_para + ';' + f'{best_auroc_mean:.4f}' + ';'
                + f'{best_auroc_std:.4f}' + '\n')

    results_fd.close()
    best_results_fd.close()
