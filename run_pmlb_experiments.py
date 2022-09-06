from sklearn.linear_model import LogisticRegression, Ridge, Lasso, RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, log_loss
from sklearn import preprocessing
from pygam.pygam import LinearGAM, LogisticGAM
from pygam import terms
from interpret.glassbox import ExplainableBoostingRegressor, ExplainableBoostingClassifier
import pandas as pd
import numpy as np
import argparse
from copy import deepcopy
from itertools import product
from functools import partial
from igann import IGANN
from pmlb import fetch_data, classification_dataset_names, regression_dataset_names
import time
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--model', type=str, required=True)

args = parser.parse_args()
task = args.task
model = args.model

if task == 'regression':
    model_pool = {
        'lasso':
           [('1e-5', Lasso(alpha=1e-5, max_iter=10000)),
            ('1e-4', Lasso(alpha=1e-4, max_iter=10000)),
            ('1e-3', Lasso(alpha=1e-3, max_iter=10000)),
            ('1e-2', Lasso(alpha=1e-2, max_iter=10000)),
            ('1e-1', Lasso(alpha=1e-1, max_iter=10000)),
            ('1e0', Lasso(alpha=1e0, max_iter=10000)),
            ('1e1', Lasso(alpha=1e1, max_iter=10000)),
            ('1e2', Lasso(alpha=1e2, max_iter=10000)),
            ('1e3', Lasso(alpha=1e3, max_iter=10000)),
            ('1e4', Lasso(alpha=1e4, max_iter=10000))],
         'ridge':
           [('1e-5', Ridge(alpha=1e-5)),
            ('1e-4', Ridge(alpha=1e-4)),
            ('1e-3', Ridge(alpha=1e-3)),
            ('1e-2', Ridge(alpha=1e-2)),
            ('1e-1', Ridge(alpha=1e-1)),
            ('1e0', Ridge(alpha=1e0)),
            ('1e1', Ridge(alpha=1e1)),
            ('1e2', Ridge(alpha=1e2)),
            ('1e3', Ridge(alpha=1e3)),
            ('1e4', Ridge(alpha=1e4))],
         'gam':
           [('5_0.2', partial(terms.s, n_splines=5, lam=0.2)),
            ('5_0.5', partial(terms.s, n_splines=5, lam=0.5)),
            ('5_0.7', partial(terms.s, n_splines=5, lam=0.7)),
            ('5_0.9', partial(terms.s, n_splines=5, lam=0.9)),
            ('10_0.2', partial(terms.s, n_splines=10, lam=0.2)),
            ('10_0.5', partial(terms.s, n_splines=10, lam=0.5)),
            ('10_0.7', partial(terms.s, n_splines=10, lam=0.7)),
            ('10_0.9', partial(terms.s, n_splines=10, lam=0.9)),
            ('15_0.2', partial(terms.s, n_splines=15, lam=0.2)),
            ('15_0.5', partial(terms.s, n_splines=15, lam=0.5)),
            ('15_0.7', partial(terms.s, n_splines=15, lam=0.7)),
            ('15_0.9', partial(terms.s, n_splines=15, lam=0.9)),
            ('20_0.2', partial(terms.s, n_splines=20, lam=0.2)),
            ('20_0.5', partial(terms.s, n_splines=20, lam=0.5)),
            ('20_0.7', partial(terms.s, n_splines=20, lam=0.7)),
            ('20_0.9', partial(terms.s, n_splines=20, lam=0.9)),
            ('25_0.2', partial(terms.s, n_splines=25, lam=0.2)),
            ('25_0.5', partial(terms.s, n_splines=25, lam=0.5)),
            ('25_0.7', partial(terms.s, n_splines=25, lam=0.7)),
            ('25_0.9', partial(terms.s, n_splines=25, lam=0.9))],
         'ebm':
           [(f'{outer_bags}_{inner_bags}_{max_bins}_{learning_rate}',
                ExplainableBoostingRegressor(outer_bags=outer_bags, inner_bags=inner_bags, max_bins=max_bins, learning_rate=learning_rate,
                    interactions=0)) for
                          outer_bags, inner_bags, max_bins, learning_rate in product(
                              [2, 4, 8, 16, 24], [0, 1, 2, 4, 8],
                              [16, 32, 64, 128, 256], [0.001, 0.005, 0.01, 0.05, 0.1])],
          'igann':
           [(f'{boost_rate}_{elm_scale}_{elm_alpha}_{init_reg}_{n_hid}', 
                IGANN(task='regression', n_estimators=10000, boost_rate=boost_rate, elm_scale=elm_scale, elm_alpha=elm_alpha,
                      init_reg=init_reg, n_hid=n_hid)) for 
                          boost_rate, elm_scale, elm_alpha, init_reg, n_hid in product(
                              [0.1, 0.3, 0.5, 0.7, 1.0], [1,2,3,5,10], 
                              [1e-7,1e-5,1e-3,1e-2,1], [1e-7,1e-5,1e-3,1e-2,1],
                              [5, 7, 10, 15, 20])],
        }
     
elif task == "classification":
    model_pool = {
        'lasso':
            [('1e-5', LogisticRegression(C=1e-5, penalty='l1', solver='liblinear')),
             ('1e-4', LogisticRegression(C=1e-4, penalty='l1', solver='liblinear')),
             ('1e-3', LogisticRegression(C=1e-3, penalty='l1', solver='liblinear')),
             ('1e-2', LogisticRegression(C=1e-2, penalty='l1', solver='liblinear')),
             ('1e-1', LogisticRegression(C=1e-1, penalty='l1', solver='liblinear')),
             ('1e0', LogisticRegression(C=1e0, penalty='l1', solver='liblinear')),
             ('1e1', LogisticRegression(C=1e1, penalty='l1', solver='liblinear')),
             ('1e2', LogisticRegression(C=1e2, penalty='l1', solver='liblinear')),
             ('1e3', LogisticRegression(C=1e3, penalty='l1', solver='liblinear')),
             ('1e4', LogisticRegression(C=1e4, penalty='l1', solver='liblinear'))],
        'ridge':
           [('1e-5', RidgeClassifier(alpha=1e-5)),
            ('1e-4', RidgeClassifier(alpha=1e-4)),
            ('1e-3', RidgeClassifier(alpha=1e-3)),
            ('1e-2', RidgeClassifier(alpha=1e-2)),
            ('1e-1', RidgeClassifier(alpha=1e-1)),
            ('1e0', RidgeClassifier(alpha=1e0)),
            ('1e1', RidgeClassifier(alpha=1e1)),
            ('1e2', RidgeClassifier(alpha=1e2)),
            ('1e3', RidgeClassifier(alpha=1e3)),
            ('1e4', RidgeClassifier(alpha=1e4))],
        'gam':
            [('5_0.2', partial(terms.s, n_splines=5, lam=0.2)),
             ('5_0.5', partial(terms.s, n_splines=5, lam=0.5)),
             ('5_0.7', partial(terms.s, n_splines=5, lam=0.7)),
             ('5_0.9', partial(terms.s, n_splines=5, lam=0.9)),
             ('10_0.2', partial(terms.s, n_splines=10, lam=0.2)),
             ('10_0.5', partial(terms.s, n_splines=10, lam=0.5)),
             ('10_0.7', partial(terms.s, n_splines=10, lam=0.7)),
             ('10_0.9', partial(terms.s, n_splines=10, lam=0.9)),
             ('15_0.2', partial(terms.s, n_splines=15, lam=0.2)),
             ('15_0.5', partial(terms.s, n_splines=15, lam=0.5)),
             ('15_0.7', partial(terms.s, n_splines=15, lam=0.7)),
             ('15_0.9', partial(terms.s, n_splines=15, lam=0.9)),
             ('20_0.2', partial(terms.s, n_splines=20, lam=0.2)),
             ('20_0.5', partial(terms.s, n_splines=20, lam=0.5)),
             ('20_0.7', partial(terms.s, n_splines=20, lam=0.7)),
             ('20_0.9', partial(terms.s, n_splines=20, lam=0.9)),
             ('25_0.2', partial(terms.s, n_splines=25, lam=0.2)),
             ('25_0.5', partial(terms.s, n_splines=25, lam=0.5)),
             ('25_0.7', partial(terms.s, n_splines=25, lam=0.7)),
             ('25_0.9', partial(terms.s, n_splines=25, lam=0.9))],
        'ebm':
           [(f'{outer_bags}_{inner_bags}_{max_bins}_{learning_rate}',
                ExplainableBoostingClassifier(outer_bags=outer_bags, inner_bags=inner_bags, max_bins=max_bins, learning_rate=learning_rate,
                    interactions=0)) for
                          outer_bags, inner_bags, max_bins, learning_rate in product(
                              [2, 4, 8, 16, 24], [0, 1, 2, 4, 8],
                              [16, 32, 64, 128, 256], [0.001, 0.005, 0.01, 0.05, 0.1])],
        'igann':
            [(f'{boost_rate}_{elm_scale}_{elm_alpha}_{init_reg}_{n_hid}',
              IGANN(task='classification', n_estimators=10000, boost_rate=boost_rate, elm_scale=elm_scale,
                    elm_alpha=elm_alpha,
                    init_reg=init_reg, n_hid=n_hid)) for
             boost_rate, elm_scale, elm_alpha, init_reg, n_hid in product(
                [0.1, 0.3, 0.5, 0.7, 1.0], [1, 2, 3, 5, 10],
                [1e-7, 1e-5, 1e-3, 1e-2, 1], [1e-7, 1e-5, 1e-3, 1e-2, 1],
                [5, 7, 10, 15, 20])],
    }
else:
    raise ValueError

drop_regression = [
    '556_analcatdata_apnea2',
    '581_fri_c3_500_25',
    '582_fri_c1_500_25',
    '583_fri_c1_1000_50',
    '584_fri_c4_500_25',
    '586_fri_c3_1000_25',
    '588_fri_c4_1000_100',
    '589_fri_c2_1000_25',
    '591_fri_c1_100_10',
    '592_fri_c4_1000_25',
    '593_fri_c1_1000_10',
    '594_fri_c2_100_5',
    '596_fri_c2_250_5',
    '597_fri_c2_500_5',
    '599_fri_c2_1000_5',
    '601_fri_c1_250_5',
    '602_fri_c3_250_10',
    '604_fri_c4_500_10',
    '605_fri_c2_250_25',
    '606_fri_c2_1000_10',
    '607_fri_c4_1000_50',
    '608_fri_c3_1000_10',
    '611_fri_c3_100_5',
    '612_fri_c1_1000_5',
    '613_fri_c3_250_5',
    '615_fri_c4_250_10',
    '616_fri_c4_500_50',
    '617_fri_c3_500_5',
    '618_fri_c3_1000_50',
    '620_fri_c1_1000_25',
    '622_fri_c2_1000_50',
    '623_fri_c4_1000_10',
    '626_fri_c2_500_50',
    '627_fri_c2_500_10',
    '628_fri_c3_1000_5',
    '631_fri_c1_500_5',
    '634_fri_c2_100_10',
    '637_fri_c1_500_50',
    '641_fri_c1_500_10',
    '643_fri_c2_500_25',
    '644_fri_c4_250_25',
    '645_fri_c3_500_50',
    '646_fri_c3_500_10',
    '647_fri_c1_250_10',
    '648_fri_c1_250_50',
    '656_fri_c1_100_5',
    '657_fri_c2_250_10',
    '658_fri_c3_250_25',
    '579_fri_c0_250_5',
     '595_fri_c0_1000_10',
     '598_fri_c0_1000_25',
     '603_fri_c0_250_50',
     '609_fri_c0_1000_5',
     '621_fri_c0_100_10',
     '624_fri_c0_100_5',
     '633_fri_c0_500_25',
     '635_fri_c0_250_10',
     '649_fri_c0_500_5',
     '650_fri_c0_500_50',
     '651_fri_c0_100_25',
     '653_fri_c0_250_25',
     '654_fri_c0_500_10']

drop_classification = [
    'GAMETES_Epistasis_2_Way_1000atts_0.4H_EDM_1_EDM_1_1',
     'GAMETES_Epistasis_2_Way_20atts_0.4H_EDM_1_1',
     'GAMETES_Epistasis_3_Way_20atts_0.2H_EDM_1_1',
     'GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_50_EDM_2_001',
     'GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_75_EDM_2_001',
     'analcatdata_boxing1',
     'clean1',
     'waveform_21',
     'solar_flare_1',
     'monk1',
     'monk3',
     'led7',
     'glass2',
     'credit_a',
     'cleve',
     'Hill_Valley_without_noise',
     'postoperative_patient_data',
     'appendicitis']

regression_dataset_names = [x for x in regression_dataset_names if x not in drop_regression]
classification_dataset_names = [x for x in classification_dataset_names if x not in drop_classification]

if task == 'regression':
    c = 0
    for regression_dataset in regression_dataset_names:
        print(f'{c}: {regression_dataset}')
        
        X, y = fetch_data(regression_dataset, return_X_y=True, local_cache_dir='data/pmlb/regression')
        if X.shape[0] > 100000:
            continue
        
        if X.shape[1] > 50:
            continue
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=1)

        scaler = preprocessing.StandardScaler().fit(X_train)

        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        scaler = preprocessing.StandardScaler().fit(y_train.reshape(-1,1))
        y_train = scaler.transform(y_train.reshape(-1,1)).squeeze()
        y_val = scaler.transform(y_val.reshape(-1,1)).squeeze()
        y_test = scaler.transform(y_test.reshape(-1,1)).squeeze()

        if model == 'igann':
            X_train = torch.from_numpy(X_train.astype(np.float32))
            y_train = torch.from_numpy(y_train.astype(np.float32))
            X_val = torch.from_numpy(X_val.astype(np.float32))
            y_val = torch.from_numpy(y_val.astype(np.float32))
            X_test = torch.from_numpy(X_test.astype(np.float32))
            y_test = torch.from_numpy(y_test.astype(np.float32))

        best_train_score = np.inf
        best_val_score = np.inf
        best_test_score = np.inf
        best_paras = ''
        
        for para_string, m in model_pool[model]:
            print(para_string)
            try:
                if model == 'ebm':
                    m = deepcopy(m)
                start = time.time()
                if model == 'lr':
                    m.fit(X_train, y_train)
                elif model == 'lasso':
                    m.fit(X_train, y_train)
                elif model == 'ridge':
                    m.fit(X_train, y_train)
                elif model == 'gam':
                    m = LinearGAM(terms.TermList(*[m(i) for i in range(X.shape[1])]))
                    m.fit(X_train, y_train)
                elif model == 'ebm':
                    m.fit(X_train, y_train)
                elif model == 'igann':
                    m.fit(X_train, y_train)
                end = time.time()
                
                val_mse = mean_squared_error(y_val, m.predict(X_val))

                if val_mse < best_val_score:
                    best_val_score = val_mse
                    best_train_score = mean_squared_error(y_train, m.predict(X_train))
                    best_test_score = mean_squared_error(y_test, m.predict(X_test))
                    best_paras = para_string
                    training_time = end - start

                if model == 'igann':
                    m._reset_state()

            except:
                continue
            
        with open(f'results/{model}_results.csv', 'a') as fd:
                fd.write(f'{model};{best_paras};{regression_dataset};{best_train_score};{best_val_score};{best_test_score};{training_time}\n')
        c += 1

elif task == 'classification':

    c = 0
    for classification_dataset in classification_dataset_names:
        print(f'{c}: {classification_dataset}')
        X, y = fetch_data(classification_dataset, return_X_y=True, local_cache_dir='data/pmlb/classification')
        if X.shape[1] > 100:
            continue

        if X.shape[0] > 100000:
            continue

        if X.shape[1] > 50:
            continue
        
        if len(np.unique(y)) != 2 or any(np.unique(y) != np.array([0,1])):
            continue

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=1)

        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    
        def map_label(label):
            if label == 1:
                return 1
            else:
                if model == 'gam':
                    return 0
                else:
                    return -1

        y_train = np.array([map_label(x) for x in y_train])
        y_val = np.array([map_label(x) for x in y_val])
        y_test = np.array([map_label(x) for x in y_test])

        if model == 'igann':
            X_train = torch.from_numpy(X_train.astype(np.float32))
            y_train = torch.from_numpy(y_train.astype(np.float32))
            X_val = torch.from_numpy(X_val.astype(np.float32))
            y_val = torch.from_numpy(y_val.astype(np.float32))
            X_test = torch.from_numpy(X_test.astype(np.float32))
            y_test = torch.from_numpy(y_test.astype(np.float32))

        best_train_score = np.inf
        best_val_score = np.inf
        best_test_score = np.inf
        best_paras = ''

        for para_string, m in model_pool[model]:
            print(para_string)
            try:
                if model == 'ebm':
                    m = deepcopy(m)
                start = time.time()
                if model == 'lasso':
                    m.fit(X_train, y_train)
                elif model == 'ridge':
                    m.fit(X_train, y_train)
                elif model == 'gam':
                    '''
                    Note: LogisticGAM from the 'pyGAM' package does not converge sometimes and raises the error: 
                    'pygam.utils.OptimizationError: PIRLS optimization has diverged.'
                    '''
                    m = LogisticGAM(terms.TermList(*[m(i) for i in range(X.shape[1])]))
                    m.fit(X_train, y_train)
                elif model == 'ebm':
                    m.fit(X_train, y_train)
                elif model == 'igann':
                    m.fit(X_train, y_train)
                end = time.time()

                val_ll = log_loss(y_val, m.predict_proba(X_val))

                if val_ll < best_val_score:
                    best_val_score = val_ll
                    best_train_score = log_loss(y_train, m.predict_proba(X_train))
                    best_test_score = log_loss(y_test, m.predict_proba(X_test))
                    best_paras = para_string
                    training_time = end - start
            
                if model == 'igann':
                    m._reset_state()

            except:
                continue

        with open(f'results/{model}_results_classification.csv', 'a') as fd:
                fd.write(f'{model};{best_paras};{classification_dataset};{best_train_score};{best_val_score};{best_test_score};{training_time}\n')
        c += 1
