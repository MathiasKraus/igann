import pandas as pd
from sklearn.datasets import load_breast_cancer

def load_breast_data():
    breast = load_breast_cancer()
    feature_names = list(breast.feature_names)
    X, y = pd.DataFrame(breast.data, columns=feature_names), breast.target
    y = y * 2 - 1
    dataset = {
        'problem': 'classification',
        'full': {
            'X': X,
            'y': y,
        },
    }
    return dataset


def load_rul_data():
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv")
    df = df.drop(['UDI', 'Product ID'], axis=1)

    y_df = df['Machine failure']
    X_df = df.drop('Machine failure', axis=1)
    y_df = y_df * 2 - 1

    dataset = {
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }

    return dataset


def load_adult_data():
    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        header=None)
    df.columns = [
        "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
        "MaritalStatus", "Occupation", "Relationship", "Race", "Gender",
        "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"
    ]
    train_cols = df.columns[0:-1]
    label = df.columns[-1]
    X_df = df[train_cols]
    y_df = df[label]
    y_df = y_df.replace({' <=50K': 0, ' >50K': 1})
    y_df = y_df.astype(int)
    y_df = y_df * 2 - 1

    dataset = {
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }

    return dataset


def load_heart_data():
    # https://www.kaggle.com/ronitf/heart-disease-uci
    df = pd.read_csv('../data/heart.csv')
    train_cols = df.columns[0:-1]
    label = df.columns[-1]
    X_df = df[train_cols]
    y_df = df[label]

    y_df = y_df * 2 - 1
    dataset = {
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }

    return dataset


def load_credit_data():
    # https://www.kaggle.com/mlg-ulb/creditcardfraud
    df = pd.read_csv('../data/creditcard.csv')
    train_cols = df.columns[0:-1]
    label = df.columns[-1]
    X_df = df[train_cols]
    y_df = df[label]
    y_df = y_df * 2 - 1
    dataset = {
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }

    return dataset


def load_telco_churn_data():
    # https://www.kaggle.com/blastchar/telco-customer-churn/downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv/1
    df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    train_cols = df.columns[1:-1]  # First column is an ID
    label = df.columns[-1]
    X_df = df[train_cols]
    X_df['TotalCharges'] = X_df['TotalCharges'].fillna('0').replace({' ': '0'}).astype(float)
    y_df = df[label]  # 'Yes, No'

    y_df = y_df.replace({'Yes': 1, 'No': -1})
    y_df = y_df.astype(int)

    dataset = {
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }

    return dataset


def load_bank_marketing_data():
    # https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
    df = pd.read_csv('../data/bank-full.csv', sep=';')
    y_df = df['y']
    X_df = df.drop('y', axis=1)

    y_df = y_df.replace({'yes': 1, 'no': -1})
    y_df = y_df.astype(int)

    dataset = {
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }

    return dataset


def load_wine_data():
    # https://www.kaggle.com/nareshbhat/wine-quality-binary-classification
    df = pd.read_csv('../data/wine.csv', sep=',')
    y_df = df['quality']
    X_df = df.drop('quality', axis=1)

    y_df = y_df.replace({'good': 1, 'bad': -1})
    y_df = y_df.astype(int)

    dataset = {
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }

    return dataset


def load_gamma_telescope_data():
    # https://archive.ics.uci.edu/ml/machine-learning-databases/magic/
    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data",
        header=None)
    df.columns = [
        "fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long",
        "fM3Trans", "fAlpha", "fDist", "class"
    ]
    y_df = df['class']
    X_df = df.drop('class', axis=1)

    y_df = y_df.replace({'g': 1, 'h': -1})
    y_df = y_df.astype(int)

    dataset = {
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }

    return dataset


def load_skin_segmentation_data():
    # https://archive.ics.uci.edu/ml/machine-learning-databases/00229/
    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt",
        sep='\t',
        header=None)
    df.columns = [
        "B", "G", "R", "class"
    ]
    y_df = df['class']
    X_df = df.drop('class', axis=1)

    y_df = y_df.replace({1: 1, 2: -1})
    y_df = y_df.astype(int)

    dataset = {
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }

    return dataset


def load_stroke_data():
    # https://www.kaggle.com/fedesoriano/stroke-prediction-dataset
    df = pd.read_csv('../data/healthcare-dataset-stroke-data.csv', sep=',')
    df['bmi'] = df['bmi'].fillna(df.bmi.mean())
    y_df = df['stroke']
    X_df = df.drop('stroke', axis=1)

    y_df = y_df.replace({1: 1, 0: -1})
    y_df = y_df.astype(int)

    dataset = {
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }

    return dataset


def load_rain_australia_data():
    # https://www.kaggle.com/fedesoriano/stroke-prediction-dataset
    df = pd.read_csv('../data/weatherAUS.csv', sep=',')
    df = df[~pd.isnull(df['RainTomorrow'])]
    y_df = df['RainTomorrow']
    X_df = df.drop('RainTomorrow', axis=1)

    X_df = X_df.fillna(X_df.mean())
    X_df['WindGustDir'].fillna('missing', inplace=True)
    X_df['WindDir9am'].fillna('missing', inplace=True)
    X_df['WindDir3pm'].fillna('missing', inplace=True)
    X_df['RainToday'].fillna('missing', inplace=True)

    y_df = y_df.replace({'Yes': 1, 'No': -1})
    y_df = y_df.astype(int)

    dataset = {
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }

    return dataset


def load_gender_recognition_data():
    # https://www.kaggle.com/primaryobjects/voicegender
    df = pd.read_csv('../data/voice.csv', sep=',')
    y_df = df['label']
    X_df = df.drop('label', axis=1)

    y_df = y_df.replace({'female': 1, 'male': -1})
    y_df = y_df.astype(int)

    dataset = {
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }

    return dataset


def load_dont_get_kicked_data():
    # https://www.kaggle.com/c/DontGetKicked/data
    df = pd.read_csv('../data/dont_get_kicked.csv', sep=',')
    df = df.drop('RefId', axis=1)

    y_df = df['IsBadBuy']
    X_df = df.drop('IsBadBuy', axis=1)

    X_df = X_df.fillna(X_df.mean())
    X_df['SubModel'].fillna('missing', inplace=True)
    X_df['Color'].fillna('missing', inplace=True)
    X_df['Transmission'].fillna('missing', inplace=True)
    X_df['Trim'].fillna('missing', inplace=True)
    X_df['WheelType'].fillna('missing', inplace=True)
    X_df['Nationality'].fillna('missing', inplace=True)
    X_df['Size'].fillna('missing', inplace=True)
    X_df['TopThreeAmericanName'].fillna('missing', inplace=True)
    X_df['PRIMEUNIT'].fillna('missing', inplace=True)
    X_df['AUCGUART'].fillna('missing', inplace=True)

    y_df = y_df.replace({1: 1, 0: -1})
    y_df = y_df.astype(int)

    dataset = {
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }

    return dataset


def load_airline_passenger_data():
    # https://www.kaggle.com/teejmahal20/airline-passenger-satisfaction
    df = pd.read_csv('../data/airline.csv', sep=',')
    df = df.drop(['Unnamed: 0', 'id'], axis=1)
    y_df = df['satisfaction']
    X_df = df.drop('satisfaction', axis=1)

    X_df = X_df.fillna(X_df.mean())

    y_df = y_df.replace({'satisfied': 1, 'neutral or dissatisfied': -1})
    y_df = y_df.astype(int)

    dataset = {
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }

    return dataset


def load_water_quality_data():
    # https://www.kaggle.com/adityakadiwal/water-potability
    df = pd.read_csv('../data/water_potability.csv', sep=',')
    y_df = df['Potability']
    X_df = df.drop('Potability', axis=1)
    X_df = X_df.fillna(X_df.mean())

    y_df = y_df.replace({1: 1, 0: -1})
    y_df = y_df.astype(int)

    dataset = {
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }

    return dataset