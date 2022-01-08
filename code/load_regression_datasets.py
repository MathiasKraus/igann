import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
import numpy as np
import glob

def check_data(X_Train, X_Val, X_Test, Y_Train, Y_Val, Y_Test, write_profile_to_file):
    import pandas_profiling
    X_Train = pd.DataFrame(X_Train, columns=['col_{}'.format(i) for i in range(X_Train.shape[1])])
    X_Val = pd.DataFrame(X_Val, columns=['col_{}'.format(i) for i in range(X_Val.shape[1])])
    X_Test = pd.DataFrame(X_Test, columns=['col_{}'.format(i) for i in range(X_Test.shape[1])])
    df = pd.concat([X_Train, X_Val, X_Test])
    df['y'] = np.hstack([Y_Train.squeeze(), Y_Val.squeeze(), Y_Test.squeeze()])
    if len(df) > 50000:
        df = df.sample(50000)
    profile = df.profile_report()
    profile.to_file(output_file=write_profile_to_file)


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

def get_base_path():
    path = '../data/regression/'
    return path

def get_dataset_1(random_state, write_profile_to_file=''):
    base_path = get_base_path()
    df=pd.read_table(base_path + "Dataset_1/machine.data",delimiter=",",names=["VENDOR","MODEL","MYCT","MMIN","MMAX","CACH","CHMIN","CHMAX","PRP","ERP"])
    
    # One hot encoding for vendors
    #df = pd.concat([df,pd.get_dummies(df['VENDOR'], prefix='VENDOR')],axis=1)
    #df.drop(['VENDOR'],axis=1, inplace=True)
    # One hot encoding for Models
    #df = pd.concat([df,pd.get_dummies(df['MODEL'], prefix='MODEL')],axis=1)
    #df.drop(['MODEL'],axis=1, inplace=True)
    
    df.drop(['VENDOR'],axis=1, inplace=True)
    df.drop(['MODEL'],axis=1, inplace=True)
    df.drop(['ERP'],axis=1, inplace=True)
    
    # Scalers
    covariate_scaler=RobustScaler()
    target_scaler=MinMaxScaler()
    
    # Scaling
    df.loc[:,df.columns!='PRP']=covariate_scaler.fit_transform(df.loc[:,df.columns!='PRP'])
    df[['PRP']]=target_scaler.fit_transform(df[['PRP']])
    
    # Create numpy arrays
    X=np.array(df.loc[:,df.columns!='PRP'])
    Y=np.ravel(np.array(df[['PRP']]))
    
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, random_state=random_state, train_size=0.8)
    X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_Train, Y_Train, random_state=random_state, train_size=0.9)
    
    if len(write_profile_to_file) > 0:
        check_data(X_Train, X_Val, X_Test, Y_Train, Y_Val, Y_Test, write_profile_to_file)
    
    return X_Train, X_Val, X_Test, Y_Train, Y_Val, Y_Test
 
def get_dataset_2(random_state, write_profile_to_file=''):
    base_path = get_base_path()
    df=pd.read_table(base_path + "Dataset_2/forestfires.csv",delimiter=",")
    
    # Label encoding of month
    df['month'].replace({'jan': 1,'feb': 2,'mar': 3,'apr': 4,'may': 5,'jun': 6,'jul': 7,'aug': 8,'sep': 9,'oct': 10,'nov': 11,'dec': 12},inplace=True)
    
    # Label encoding of day
    df['day'].replace({'mon': 1,'tue': 2,'wed': 3,'thu': 4,'fri': 5,'sat': 6,'sun': 7},inplace=True)
    
    # Log transform of area as recommended
    df['area']=np.log(1+df['area'])
    
    # Scalers
    covariate_scaler=RobustScaler()
    target_scaler=MinMaxScaler()
    
    # Scaling
    df.loc[:,df.columns!='area']=covariate_scaler.fit_transform(df.loc[:,df.columns!='area'])
    df[['area']]=target_scaler.fit_transform(df[['area']])
    
    # Create numpy arrays
    X=np.array(df.loc[:,df.columns!='area'])
    Y=np.ravel(np.array(df[['area']]))
    
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, random_state=random_state, train_size=0.8)
    X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_Train, Y_Train, random_state=random_state, train_size=0.9)
    
    if len(write_profile_to_file) > 0:
        check_data(X_Train, X_Val, X_Test, Y_Train, Y_Val, Y_Test, write_profile_to_file)
    
    return X_Train, X_Val, X_Test, Y_Train, Y_Val, Y_Test
    
def get_dataset_3(random_state, write_profile_to_file=''):
    base_path = get_base_path()
    file_path = base_path + "Dataset_3/stock portfolio performance data set.xlsx"
    sheets = ['1st period', '2nd period', '3rd period', '4th period']
    df=pd.concat(pd.read_excel(file_path, sheet_name=sheet, header=1) for sheet in sheets)
    
    # Keep inputs and normalized annual return as target
    df = df[[' Large B/P ', ' Large ROE ', ' Large S/P ', ' Large Return Rate in the last quarter ', 
             ' Large Market Value ', ' Small systematic Risk', 'Annual Return.1']]
    
    # Scalers
    covariate_scaler=RobustScaler()
    target_scaler=MinMaxScaler()
    
    # Scaling
    df.loc[:,df.columns!='Annual Return.1']=covariate_scaler.fit_transform(df.loc[:,df.columns!='Annual Return.1'])
    df[['Annual Return.1']]=target_scaler.fit_transform(df[['Annual Return.1']])
    
    # Create numpy arrays
    X=np.array(df.loc[:,df.columns!='Annual Return.1'])
    Y=np.ravel(np.array(df[['Annual Return.1']]))
    
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, random_state=random_state, train_size=0.8)
    X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_Train, Y_Train, random_state=random_state, train_size=0.9)
    
    if len(write_profile_to_file) > 0:
        check_data(X_Train, X_Val, X_Test, Y_Train, Y_Val, Y_Test, write_profile_to_file)
    
    return X_Train, X_Val, X_Test, Y_Train, Y_Val, Y_Test

def get_dataset_4(random_state, write_profile_to_file=''):
    base_path = get_base_path()
    df=pd.read_table(base_path + "Dataset_4/yacht_hydrodynamics.data",delimiter=" ",names=["longitude","prismatic_coefficient","length_displacement","beam_draught_ratio","length_beam_ratio","froude_number","residuary_resistance"])
    
    # Delete prismatic_coefficient due to missing values
    df.drop(['prismatic_coefficient'],axis=1, inplace=True)
    
    # Scalers
    covariate_scaler=RobustScaler()
    target_scaler=MinMaxScaler()
    
    # Scaling
    df.loc[:,df.columns!='residuary_resistance']=covariate_scaler.fit_transform(df.loc[:,df.columns!='residuary_resistance'])
    df[['residuary_resistance']]=target_scaler.fit_transform(df[['residuary_resistance']])
    
    # Create numpy arrays
    X=np.array(df.loc[:,df.columns!='residuary_resistance'])
    Y=np.ravel(np.array(df[['residuary_resistance']]))
    
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, random_state=random_state, train_size=0.8)
    X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_Train, Y_Train, random_state=random_state, train_size=0.9)
    
    if len(write_profile_to_file) > 0:
        check_data(X_Train, X_Val, X_Test, Y_Train, Y_Val, Y_Test, write_profile_to_file)
    
    return X_Train, X_Val, X_Test, Y_Train, Y_Val, Y_Test

def get_dataset_5(random_state, write_profile_to_file=''):
    base_path = get_base_path()
    df=pd.read_table(base_path + "Dataset_5/dataset_Facebook.csv",delimiter=";")
    
    # Total interactions = comment + like + share
    df.drop(['comment'],axis=1, inplace=True)
    df.drop(['like'],axis=1, inplace=True)
    df.drop(['share'],axis=1, inplace=True)
    
    # One hot encoding of Type
    df = pd.concat([df,pd.get_dummies(df['Type'], prefix='Type')],axis=1)
    df.drop(['Type'],axis=1, inplace=True)
    
    # Drop Nans    
    df = df.dropna()
    
    # Scalers
    covariate_scaler=RobustScaler()
    target_scaler=MinMaxScaler()
    
    # Scaling
    df.loc[:,df.columns!='Total Interactions']=covariate_scaler.fit_transform(df.loc[:,df.columns!='Total Interactions'])
    df[['Total Interactions']]=target_scaler.fit_transform(df[['Total Interactions']])
    
    # Create numpy arrays
    X=np.array(df.loc[:,df.columns!='Total Interactions'])
    Y=np.ravel(np.array(df[['Total Interactions']]))
    
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, random_state=random_state, train_size=0.8)
    X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_Train, Y_Train, random_state=random_state, train_size=0.9)
    
    if len(write_profile_to_file) > 0:
        check_data(X_Train, X_Val, X_Test, Y_Train, Y_Val, Y_Test, write_profile_to_file)
    
    return X_Train, X_Val, X_Test, Y_Train, Y_Val, Y_Test

def get_dataset_6(random_state, write_profile_to_file=''):
    base_path = get_base_path()
    df=pd.read_table(base_path + "Dataset_6/Residential-Building-Data-Set.csv",delimiter=";")
    
    # Filter out unused features
    df.drop(['START YEAR', 'START QUARTER', 'COMPLETION YEAR', 'COMPLETION QUARTER'], axis=1, inplace=True)
    
    # Use time lag 4 that shows best results according to https://ascelibrary.org/doi/pdf/10.1061/%28ASCE%29CO.1943-7862.0001570
    df.drop(['V-{}'.format(x) for x in range(11,30)], axis=1, inplace=True)
    df.drop(['V-{}.{}'.format(x, y) for x in range(11,30) for y in [1, 2, 3]], axis=1, inplace=True)
    
    # V-9 = construction costs , V-10 = sales price ==> predict only sales price drop the other output variable
    df.drop(['V-9'],axis=1, inplace=True)
    
    # Scalers
    covariate_scaler=RobustScaler()
    target_scaler=MinMaxScaler()
    
    # Scaling
    df.loc[:,df.columns!='V-10']=covariate_scaler.fit_transform(df.loc[:,df.columns!='V-10'])
    df[['V-10']]=target_scaler.fit_transform(df[['V-10']])
    
    # Create numpy arrays
    X=np.array(df.loc[:,df.columns!='V-10'])
    Y=np.ravel(np.array(df[['V-10']]))
    
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, random_state=random_state, train_size=0.8)
    X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_Train, Y_Train, random_state=random_state, train_size=0.9)
    
    if len(write_profile_to_file) > 0:
        check_data(X_Train, X_Val, X_Test, Y_Train, Y_Val, Y_Test, write_profile_to_file)
    
    return X_Train, X_Val, X_Test, Y_Train, Y_Val, Y_Test

def get_dataset_7(random_state, write_profile_to_file=''):
    base_path = get_base_path()
    df=pd.read_table(base_path + "Dataset_7/Real_estate_valuation_data_set.csv",delimiter=";")
    
    # Drop numbering
    df.drop(['No'],axis=1, inplace=True)
    
    # Scalers
    covariate_scaler=RobustScaler()
    target_scaler=MinMaxScaler()
    
    # Scaling
    df.loc[:,df.columns!='Y house price of unit area']=covariate_scaler.fit_transform(df.loc[:,df.columns!='Y house price of unit area'])
    df[['Y house price of unit area']]=target_scaler.fit_transform(df[['Y house price of unit area']])
    
    # Create numpy arrays
    X=np.array(df.loc[:,df.columns!='Y house price of unit area'])
    Y=np.ravel(np.array(df[['Y house price of unit area']]))
    
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, random_state=random_state, train_size=0.8)
    X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_Train, Y_Train, random_state=random_state, train_size=0.9)
    
    if len(write_profile_to_file) > 0:
        check_data(X_Train, X_Val, X_Test, Y_Train, Y_Val, Y_Test, write_profile_to_file)
    
    return X_Train, X_Val, X_Test, Y_Train, Y_Val, Y_Test

def get_dataset_8(random_state, write_profile_to_file=''):
    base_path = get_base_path()
    df=pd.read_table(base_path + "Dataset_8/qsar_fish_toxicity.csv",delimiter=";",names=["CIC0","SM1_Dz","GATS1i","NdsCH","NdssC","MLOGP","quant_response"])
    
    # Scalers
    covariate_scaler=RobustScaler()
    target_scaler=MinMaxScaler()
    
    # Scaling
    df.loc[:,df.columns!='quant_response']=covariate_scaler.fit_transform(df.loc[:,df.columns!='quant_response'])
    df[['quant_response']]=target_scaler.fit_transform(df[['quant_response']])
    
    # Create numpy arrays
    X=np.array(df.loc[:,df.columns!='quant_response'])
    Y=np.ravel(np.array(df[['quant_response']]))
    
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, random_state=random_state, train_size=0.8)
    X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_Train, Y_Train, random_state=random_state, train_size=0.9)
    
    if len(write_profile_to_file) > 0:
        check_data(X_Train, X_Val, X_Test, Y_Train, Y_Val, Y_Test, write_profile_to_file)
    
    return X_Train, X_Val, X_Test, Y_Train, Y_Val, Y_Test

def get_dataset_9(random_state, write_profile_to_file=''):
    base_path = get_base_path()
    df=pd.read_table(base_path + "Dataset_9/qsar_aquatic_toxicity.csv",delimiter=";",names=["TPSA","SAacc","H-050","MLOGP","RDCHI","GATS1p","nN","C-040","quant_response"])
    
    # Scalers
    covariate_scaler=RobustScaler()
    target_scaler=MinMaxScaler()
    
    # Scaling
    df.loc[:,df.columns!='quant_response']=covariate_scaler.fit_transform(df.loc[:,df.columns!='quant_response'])
    df[['quant_response']]=target_scaler.fit_transform(df[['quant_response']])
    
    # Create numpy arrays
    X=np.array(df.loc[:,df.columns!='quant_response'])
    Y=np.ravel(np.array(df[['quant_response']]))
    
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, random_state=random_state, train_size=0.8)
    X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_Train, Y_Train, random_state=random_state, train_size=0.9)
    
    if len(write_profile_to_file) > 0:
        check_data(X_Train, X_Val, X_Test, Y_Train, Y_Val, Y_Test, write_profile_to_file)
    
    return X_Train, X_Val, X_Test, Y_Train, Y_Val, Y_Test


def load_crimes_data(random_state):
    # https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime
    base_path = get_base_path()
    df = pd.read_csv(base_path + 'Dataset_10/communities.data', sep=',')
    X_df = df.drop(['ViolentCrimesPerPop', 'state', 'county', 'community', 'communityname string', 'fold'], axis=1)

    X_df = X_df.replace("?", "")
    X_df = preprocess_columns(X_df)

    X_df = X_df.drop(['LemasGangUnitDeploy', 'NumKindsDrugsSeiz'], axis=1)

    y_df = df['ViolentCrimesPerPop']
    pt = PowerTransformer()
    y_np = pt.fit_transform(y_df.to_numpy().reshape(-1, 1))
    y_df = pd.DataFrame(data=y_np, columns=["y"])

    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_df, y_df, random_state=random_state, train_size=0.8)
    X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_Train, Y_Train, random_state=random_state, train_size=0.9)

    return X_Train, X_Val, X_Test, np.squeeze(Y_Train), np.squeeze(Y_Val), np.squeeze(Y_Test)


def load_bike_sharing_data(random_state):
    # https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset
    base_path = get_base_path()
    df = pd.read_csv(base_path + 'Dataset_11/bike.csv', sep=',')
    X_df = df.drop(['instant', 'dteday', 'cnt', 'casual', 'registered'], axis=1)

    X_df['season'] = X_df['season'].astype(str)
    X_df['yr'] = X_df['yr'].astype(str)
    X_df['holiday'] = X_df['holiday'].astype(str)
    X_df['weekday'] = X_df['weekday'].astype(str)
    X_df['workingday'] = X_df['workingday'].astype(str)
    X_df['weathersit'] = X_df['weathersit'].astype(str)

    X_df = preprocess_columns(X_df)

    y_df = df['cnt']
    pt = PowerTransformer()
    y_np = pt.fit_transform(y_df.to_numpy().reshape(-1, 1))
    y_df = pd.DataFrame(data=y_np, columns=["y"])

    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_df, y_df, random_state=random_state, train_size=0.8)
    X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_Train, Y_Train, random_state=random_state, train_size=0.9)

    return X_Train, X_Val, X_Test, np.squeeze(Y_Train), np.squeeze(Y_Val), np.squeeze(Y_Test)


def load_california_housing_data(random_state):
    # https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html
    base_path = get_base_path()
    df = pd.read_csv(base_path + 'Dataset_12/cal_housing.data', sep=',')
    X_df = df.drop(['medianHouseValue'], axis=1)

    X_df = preprocess_columns(X_df)

    y_df = df['medianHouseValue']
    pt = PowerTransformer()
    y_np = pt.fit_transform(y_df.to_numpy().reshape(-1, 1))
    y_df = pd.DataFrame(data=y_np, columns=["y"])

    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_df, y_df, random_state=random_state, train_size=0.8)
    X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_Train, Y_Train, random_state=random_state, train_size=0.9)

    return X_Train, X_Val, X_Test, np.squeeze(Y_Train), np.squeeze(Y_Val), np.squeeze(Y_Test)


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
    elif str_id == '10':
        X_train, X_Val, X_test, Y_train, Y_Val, Y_test = load_crimes_data(random_state)
    elif str_id == '11':
        X_train, X_Val, X_test, Y_train, Y_Val, Y_test = load_bike_sharing_data(random_state)
    elif str_id == '12':
        X_train, X_Val, X_test, Y_train, Y_Val, Y_test = load_california_housing_data(random_state)
    else:
        raise ValueError('Unknown dataset')
        
    X_train = np.vstack([X_train, X_Val])
    Y_train = np.hstack([Y_train, Y_Val])
    return X_train, X_test, Y_train, Y_test