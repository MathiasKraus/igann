"""
Copyright Nico Hambauer, Mathias Kraus, Sven Weinzierl, Sandra Zilker Patrick Zschech
No License
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split


def preprocess_columns(df):
    """
    Variables mit missing values >50% rauswerfen
    Num. Variables mit missing values per mean ersetzen
    Cat. Variables mit missing vlaues
    Kategoriale Variablen mit mehr als 25 Ausprägungen rauswerfen
    :return:
    """

    # remove features because of missing values
    mv_cols = df.columns[df.isnull().sum() / len(df) > 0.5]
    df.drop(mv_cols, axis=1, inplace=True)

    # remove cat features because of num values
    cols = df.columns
    num_cols = df._get_numeric_data().columns
    cat_cols = list(set(cols) - set(num_cols))

    if len(cat_cols) > 0:
        for cat_col in cat_cols:
            if len(df[cat_col].unique()) > 25:
                df.drop(cat_col, axis=1, inplace=True)

    # handle missing values
    cols = df.columns
    num_cols = df._get_numeric_data().columns
    cat_cols = list(set(cols) - set(num_cols))

    if len(cat_cols) > 0:
        for cat_col in cat_cols:
            df[cat_col] = df[cat_col].fillna(-1)

    if len(num_cols) > 0:
        for num_col in num_cols:
            df[num_col] = df[num_col].fillna(df[num_col].mean())

    return df


def load_water_quality_data(random_state):
    # https://www.kaggle.com/adityakadiwal/water-potability
    df = pd.read_csv('../data/water_potability.csv', sep=',')
    y_df = df['Potability']
    X_df = df.drop('Potability', axis=1)
    X_df = preprocess_columns(X_df)

    y_df = y_df.astype(int)
    
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_df, y_df, random_state=random_state, train_size=0.8)
    X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_Train, Y_Train, random_state=random_state, train_size=0.9)

    return X_Train, X_Val, X_Test, Y_Train, Y_Val, Y_Test


def load_stroke_data(random_state):
    # https://www.kaggle.com/fedesoriano/stroke-prediction-dataset
    df = pd.read_csv('../data/healthcare_dataset_stroke_data.csv', sep=',')
    y_df = df['stroke']
    X_df = df.drop('stroke', axis=1)

    X_df['hypertension'] = X_df['hypertension'].replace({1: "Yes", 0: "No"})
    X_df['heart_disease'] = X_df['heart_disease'].replace({1: "Yes", 0: "No"})

    cols = X_df.columns
    num_cols = X_df._get_numeric_data().columns
    cat_cols = list(set(cols) - set(num_cols))
    num_cols = [num_col for num_col in num_cols if num_col in ['age', 'avg_glucose_level', 'bmi']]

    X_df = X_df[cat_cols+num_cols]
    X_df = preprocess_columns(X_df)

    y_df = y_df.astype(int)

    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_df, y_df, random_state=random_state, train_size=0.8)
    X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_Train, Y_Train, random_state=random_state, train_size=0.9)

    return X_Train, X_Val, X_Test, Y_Train, Y_Val, Y_Test


def load_telco_churn_data(random_state):
    # https://www.kaggle.com/blastchar/telco-customer-churn/downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv/1
    df = pd.read_csv('../data/WA_Fn_UseC__Telco_Customer_Churn.csv')
    y_df = df['Churn']
    X_df = df.drop(['Churn', 'customerID'], axis=1)

    X_df['SeniorCitizen'] = X_df['SeniorCitizen'].replace({1: "Yes", 0: "No"})
    X_df['TotalCharges'] = pd.to_numeric(X_df['TotalCharges'].replace(" ", ""))

    cols = X_df.columns
    num_cols = X_df._get_numeric_data().columns
    cat_cols = list(set(cols) - set(num_cols))
    num_cols = [num_col for num_col in num_cols if num_col in ['tenure', 'MonthlyCharges', 'TotalCharges']]

    X_df = X_df[cat_cols + num_cols]
    X_df = preprocess_columns(X_df)

    y_df = y_df.replace({'Yes': 1, 'No': 0})
    y_df = y_df.astype(int)

    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_df, y_df, random_state=random_state, train_size=0.8)
    X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_Train, Y_Train, random_state=random_state, train_size=0.9)

    return X_Train, X_Val, X_Test, Y_Train, Y_Val, Y_Test


def load_fico_data(random_state):
    # https://community.fico.com/s/explainable-machine-learning-challenge?tabset-158d9=3
    df = pd.read_csv('../data/fico_heloc_dataset_v1.csv')
    X_df = df.drop(['RiskPerformance'], axis=1)

    X_df['MaxDelq2PublicRecLast12M'] = X_df['MaxDelq2PublicRecLast12M'].astype(str)
    X_df['MaxDelqEver'] = X_df['MaxDelqEver'].astype(str)

    cols = X_df.columns
    num_cols = X_df._get_numeric_data().columns
    cat_cols = list(set(cols) - set(num_cols))
    cat_cols = [cat_col for cat_col in cat_cols if cat_col in ['MaxDelq2PublicRecLast12M', 'MaxDelqEver']]

    X_df = X_df[cat_cols+num_cols.tolist()]
    X_df = preprocess_columns(X_df)

    y_df = df['RiskPerformance']
    y_df = y_df.replace({'Good': 1, 'Bad': 0})
    y_df = y_df.astype(int)

    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_df, y_df, random_state=random_state, train_size=0.8)
    X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_Train, Y_Train, random_state=random_state, train_size=0.9)

    return X_Train, X_Val, X_Test, Y_Train, Y_Val, Y_Test


def load_bank_marketing_data(random_state):
    # https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
    df = pd.read_csv('../data/bank_full.csv', sep=';')
    y_df = df['y']
    X_df = df.drop('y', axis=1)

    cols = X_df.columns
    num_cols = X_df._get_numeric_data().columns
    cat_cols = list(set(cols) - set(num_cols))
    num_cols = [num_col for num_col in num_cols if num_col in ['age', 'duration', 'campaign', 'pdays', 'previous']]

    X_df = X_df[cat_cols + num_cols]
    X_df = preprocess_columns(X_df)

    y_df = y_df.replace({'yes': 1, 'no': 0})
    y_df = y_df.astype(int)

    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_df, y_df, random_state=random_state, train_size=0.8)
    X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_Train, Y_Train, random_state=random_state, train_size=0.9)

    return X_Train, X_Val, X_Test, Y_Train, Y_Val, Y_Test


def load_adult_data(random_state):
    df = pd.read_csv('../data/adult_census_income.csv')
    X_df = df.drop(['income'], axis=1)

    cols = X_df.columns
    num_cols = X_df._get_numeric_data().columns
    cat_cols = list(set(cols) - set(num_cols))
    num_cols = [num_col for num_col in num_cols if num_col in ['age', 'fnlwgt', 'education.num',
                                                               'capital.gain', 'capital.loss', 'hours.per.week']]

    X_df = X_df[cat_cols + num_cols]
    X_df = preprocess_columns(X_df)

    y_df = df["income"]
    y_df = y_df.replace({' <=50K': 0, ' >50K': 1})
    y_df = y_df.astype(int)

    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_df, y_df, random_state=random_state, train_size=0.8)
    X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_Train, Y_Train, random_state=random_state, train_size=0.9)

    return X_Train, X_Val, X_Test, Y_Train, Y_Val, Y_Test


def load_airline_passenger_data(random_state):
    # https://www.kaggle.com/teejmahal20/airline-passenger-satisfaction
    df = pd.read_csv('../data/airline_train.csv', sep=',')
    y_df = df['satisfaction']
    X_df = df.drop(['Unnamed: 0', 'id', 'satisfaction'], axis=1)

    cols = X_df.columns
    num_cols = X_df._get_numeric_data().columns
    cat_cols = list(set(cols) - set(num_cols))
    cat_cols = [cat_col for cat_col in cat_cols if cat_col in ['Gender', 'Customer Type', 'Type of Travel', 'Class']]

    X_df = X_df[cat_cols + num_cols.tolist()]
    X_df = preprocess_columns(X_df)

    y_df = y_df.replace({'satisfied': 1, 'neutral or dissatisfied': 0})
    y_df = y_df.astype(int)

    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_df, y_df, random_state=random_state, train_size=0.8)
    X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_Train, Y_Train, random_state=random_state, train_size=0.9)

    return X_Train, X_Val, X_Test, Y_Train, Y_Val, Y_Test



def get_dataset(str_id, random_state, write_profile_to_file=''):
    if str_id == '1':
        X_train, X_Val, X_test, Y_train, Y_Val, Y_test = get_dataset_1(random_state, write_profile_to_file)
    elif str_id == '2':
        X_train, X_Val, X_test, Y_train, Y_Val, Y_test = get_dataset_2(random_state, write_profile_to_file)
    elif str_id == '3':
        X_train, X_Val, X_test, Y_train, Y_Val, Y_test = get_dataset_3(random_state, write_profile_to_file)
    elif str_id == '4':
        X_train, X_Val, X_test, Y_train, Y_Val, Y_test = get_dataset_4(random_state, write_profile_to_file)
    elif str_id == '5':
        X_train, X_Val, X_test, Y_train, Y_Val, Y_test = get_dataset_5(random_state, write_profile_to_file)
    elif str_id == '6':
        X_train, X_Val, X_test, Y_train, Y_Val, Y_test = get_dataset_6(random_state, write_profile_to_file)
    elif str_id == '7':
        X_train, X_Val, X_test, Y_train, Y_Val, Y_test = get_dataset_7(random_state, write_profile_to_file)
    elif str_id == '8':
        X_train, X_Val, X_test, Y_train, Y_Val, Y_test = get_dataset_8(random_state, write_profile_to_file)
    elif str_id == '9':
        X_train, X_Val, X_test, Y_train, Y_Val, Y_test = get_dataset_9(random_state, write_profile_to_file)
    else:
        raise ValueError('Unknown dataset')
        
    X_train = np.vstack([X_train, X_Val])
    Y_train = np.hstack([Y_train, Y_Val])
    return X_train, X_test, Y_train, Y_test
