import numpy as np
import pandas as pd

class Dataset:
    def __init__(self, dataset_id: int):
        self.name = f'churn_{dataset_id}'
        """ name of the dataset"""
        self.problem = None
        """ classification or regression """
        self.X = None
        """ X data frame """
        self.y = None
        """ y data frame """
        self.labels = None
        """ discrete label values of the dataset: classification: [0, 1]"""
        self.target_names = None
        """ name of the label: classification: ['Negative Class', 'Positive Class'], regression: ['ValueName'] """
        self.numerical_cols = None
        """ list of numerical columns which are selected after preprocessing"""
        self.categorical_cols = None
        """ list of categorical feature names which are selected after preprocessing """
        # Now load all these variables
        self._load_churn_data(dataset_id)

    def _impute_missing_value(self, df):
        """
        Drop variables with missing values >50%.
        Replace missing numerical variable values by mean.
        Replace missing categorical variable values by -1.
        Drop categorical columns with more than 25 distinct values.
        :return:
        """

        assert len(self.numerical_cols) + len(self.categorical_cols) > 0, \
            "Dataframe columns must be specified in load_datasets.py in order to preprocess them."

        # Ensure there are no empty string in numerical columns and encode as float
        df.loc[:, self.numerical_cols] = df.loc[:, self.numerical_cols].replace({'': np.nan, ' ': np.nan})

        # select columns with more than 50 % missing values
        incomplete_cols = df.columns[df.isnull().sum() / len(df) > 0.5]
        # select categorical_cols with more than 25 unique values
        detailed_cols = df[self.categorical_cols].nunique()[df[self.categorical_cols].nunique() > 25].index.tolist()

        numerical_cols = list(set(self.numerical_cols) - set(incomplete_cols))
        self.categorical_cols = list(set(self.categorical_cols) - set(incomplete_cols) - set(detailed_cols))

        df = df.loc[:, numerical_cols + self.categorical_cols]

        # For categorical columns with values: fill n/a-values with -1.
        if len(self.categorical_cols) > 0:
            for categorical_col in self.categorical_cols:
                df.loc[:, categorical_col] = df.loc[:, categorical_col].fillna('unknown')
                df.loc[:, categorical_col] = df.loc[:, categorical_col].astype('category')

        # For numerical columns with values: fill n/a-values with mean.
        if len(numerical_cols) > 0:
            for num_col in numerical_cols:
                df.loc[:, num_col] = pd.to_numeric(df.loc[:, num_col], errors='coerce')
                df.loc[:, num_col] = df.loc[:, num_col].fillna(df.loc[:, num_col].mean())

        return df

    def _load_churn_data(self, i: int):
        # first row is the header, and the first column is the index
        df = pd.read_csv(f'data2/churn_{i}.csv', sep=',', header=0, index_col=0)
        # numerical columns are identified by type float and categorical are identified by type int or object and are of kind o

        self.y = df['dependent']
        df = df.drop('dependent', axis=1)

        self.numerical_cols = df.select_dtypes(include=['float']).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['int64', 'int', 'object']).columns.tolist()

        self.X = self._impute_missing_value(df)

        # replace the label values cl0 and cl1 with 0 and 1
        self.y = self.y.replace({'cl0': 0, 'cl1': 1})
        self.y = self.y.astype(int)

        self.problem = 'classification'

        self.labels = [0, 1]
        self.target_names = ['cl0', 'cl1']