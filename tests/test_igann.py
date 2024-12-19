from igann import igann
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images

from sklearn.datasets import load_breast_cancer, load_diabetes, make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, mean_squared_error

import numpy as np

import torch

#def test_sparse_igann():
#    X, y = make_regression(100000, 10, n_informative=3, random_state=0)
#    y = (y - y.mean()) / y.std()
#    m =  igann.IGANN(task='regression', n_estimators=1000, sparse=10)
#    m.fit(pd.DataFrame(X), y)
#    assert len(m.feature_names) < 7

def test_classification_train_no_interaction_pd_df():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    scaler = StandardScaler()

    X_names = X.columns

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train, columns=X_names)
    X_test = pd.DataFrame(X_test, columns=X_names)

    model = igann.IGANN() # interactions=0)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    f1 = f1_score(y_test, preds)

    assert (f1 > 0.94)
    # f1 above 0.98 in prior tests

# def test_classification_train_find_interactions_pd_df():
#     X, y = load_breast_cancer(return_X_y=True, as_frame=True)

#     scaler = StandardScaler()

#     X_names = X.columns

#     X_train, X_test, y_train, y_test = train_test_split(X, y)

#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     X_train = pd.DataFrame(X_train, columns=X_names)
#     X_test = pd.DataFrame(X_test, columns=X_names)

#     model = igann.IGANN(interactions=2)

#     model.fit(X_train, y_train)
#     preds = model.predict(X_test)

#     f1 = f1_score(y_test, preds)

#     assert (f1 > 0.94)
#     # f1 above 0.98 in prior tests

# def test_classification_train_no_interaction_np_array():
#     X, y = load_breast_cancer(return_X_y=True, as_frame=True)

#     scaler = StandardScaler()

#     X_train, X_test, y_train, y_test = train_test_split(X, y)

#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     model = igann.IGANN() #interactions=0)

#     model.fit(X_train, y_train.to_numpy())
#     preds = model.predict(X_test)

#     f1 = f1_score(y_test.to_numpy(), preds)

#     assert (f1 > 0.94)
#     # f1 above 0.98 in prior tests

# def test_classification_train_find_interactions_np_array():
#     X, y = load_breast_cancer(return_X_y=True, as_frame=True)

#     scaler = StandardScaler()

#     X_train, X_test, y_train, y_test = train_test_split(X, y)

#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     model = igann.IGANN(interactions=2)

#     model.fit(X_train, y_train.to_numpy())
#     preds = model.predict(X_test)

#     f1 = f1_score(y_test.to_numpy(), preds)

#     assert (f1 > 0.94)
#     # f1 above 0.98 in prior tests
    
# def test_classification_predict_proba_no_interaction_np_array():
#     X, y = load_breast_cancer(return_X_y=True, as_frame=True)

#     scaler = StandardScaler()

#     X_train, X_test, y_train, y_test = train_test_split(X, y)

#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     model = igann.IGANN() # interactions=0)

#     model.fit(X_train, y_train.to_numpy())

#     preds = model.predict_proba(X_test)

#     max_indices = np.argmax(preds, axis=1)
#     f1 = f1_score(y_test.to_numpy(), max_indices)

#     assert (f1 > 0.94)
#     # f1 above 0.97 in prior tests

def test_classification_predict_proba_no_interaction_pd_df():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    scaler = StandardScaler()

    X_names = X.columns

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train, columns=X_names)
    X_test = pd.DataFrame(X_test, columns=X_names)

    model = igann.IGANN() # interactions=0)

    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)

    max_indices = np.argmax(preds, axis=1)
    f1 = f1_score(y_test, max_indices)

    assert (f1 > 0.94)
    # f1 above 0.97 in prior tests

def test_classification_plot_single():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    scaler = StandardScaler()

    X_names = X.columns

    X = scaler.fit_transform(X)

    X = pd.DataFrame(X, columns=X_names)

    model = igann.IGANN() #interactions=0)

    model.fit(X, y)

    model.plot_single()

# def test_classification_plot_interactions():
#     X, y = load_breast_cancer(return_X_y=True, as_frame=True)

#     scaler = StandardScaler()

#     X_names = X.columns

#     X = scaler.fit_transform(X)

#     X = pd.DataFrame(X, columns=X_names)

#     model = igann.IGANN(interactions=2)

#     model.fit(X, y)

#     model.plot_interactions()

def test_classification_plot_learning():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    scaler = StandardScaler()

    X_names = X.columns

    X = scaler.fit_transform(X)

    X = pd.DataFrame(X, columns=X_names)

    model = igann.IGANN() # interactions=2)

    model.fit(X, y)
    
    model.plot_learning()

def test_regression_train_no_interaction_pd_df():
    X, y = load_diabetes(return_X_y=True, as_frame=True)

    scaler = StandardScaler()

    X_names = X.columns

    kf = KFold(n_splits=8, shuffle=True)
    mse = []

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        y_test = (y_test - y_train.mean()) / y_train.std()
        y_train = (y_train - y_train.mean()) / y_train.std()

        X_train = pd.DataFrame(X_train, columns=X_names)
        X_test = pd.DataFrame(X_test, columns=X_names)

        model = igann.IGANN(task='regression') #, interactions=0)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mse.append(mean_squared_error(y_test, preds))

    assert (np.mean(mse) < 0.6)
    # mse below 3100 in prior tests

# def test_regression_train_find_interactions_pd_df():
#     X, y = load_diabetes(return_X_y=True, as_frame=True)

#     scaler = StandardScaler()

#     X_names = X.columns

#     X_train, X_test, y_train, y_test = train_test_split(X, y)

#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     X_train = pd.DataFrame(X_train, columns=X_names)
#     X_test = pd.DataFrame(X_test, columns=X_names)

#     model = igann.IGANN(task='regression', interactions=2)

#     model.fit(X_train, y_train)
#     preds = model.predict(X_test)

#     mse = mean_squared_error(y_test, preds)

#     assert (mse < 3300)
#     # mse below 3100 in prior tests


# def test_regression_train_no_interaction_np_array():
#     X, y = load_diabetes(return_X_y=True, as_frame=True)
#     scaler = StandardScaler()

#     X_train, X_test, y_train, y_test = train_test_split(X, y)

#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     model = igann.IGANN(task='regression') #, interactions=0)

#     model.fit(X_train, y_train)
#     preds = model.predict(X_test)

#     mse = mean_squared_error(y_test, preds)

#     assert (mse < 3300)
#     # mse below 3100 in prior tests

# def test_regression_train_find_interactions_np_array():
#     X, y = load_diabetes(return_X_y=True, as_frame=True)
#     scaler = StandardScaler()

#     X_train, X_test, y_train, y_test = train_test_split(X, y)

#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     model = igann.IGANN(task='regression', interactions=2)

#     model.fit(X_train, y_train)
#     preds = model.predict(X_test)

#     mse = mean_squared_error(y_test, preds)

#     assert (mse < 3300)
#     # mse below 3100 in prior tests

def test_regression_plot_single():
    X, y = load_diabetes(return_X_y=True, as_frame=True)

    scaler = StandardScaler()

    X_names = X.columns

    X = scaler.fit_transform(X)
    y = (y - y.mean()) / y.std()
    
    X = pd.DataFrame(X, columns=X_names)

    model = igann.IGANN(task='regression') #, interactions=0)

    model.fit(X, y)

    model.plot_single()
    

# def test_regression_plot_interactions():
#     X, y = load_diabetes(return_X_y=True, as_frame=True)

#     scaler = StandardScaler()

#     X_names = X.columns

#     X = scaler.fit_transform(X)

#     X = pd.DataFrame(X, columns=X_names)

#     model = igann.IGANN(task='regression', interactions=2)

#     model.fit(X, y)

#     model.plot_interactions()

def test_regression_plot_learning():
    X, y = load_diabetes(return_X_y=True, as_frame=True)

    scaler = StandardScaler()

    X_names = X.columns

    X = scaler.fit_transform(X)
    y = (y - y.mean()) / y.std()
    
    X = pd.DataFrame(X, columns=X_names)

    model = igann.IGANN(task='regression') #, interactions=2)

    model.fit(X, y)

    model.plot_learning()

'''
def test_classification_plot_single_w_baseline():
   X, y = load_breast_cancer(return_X_y=True, as_frame=True)

   scaler = StandardScaler()

   X_names = X.columns

   X = scaler.fit_transform(X)

   X = pd.DataFrame(X, columns=X_names)

   model = igann.IGANN(random_state=42) # interactions=0, 

   model.fit(X, y)

   model.plot_single()

   baseline = "tests/baseline/baseline_class_plot_single.png"

   path = "temp_class_plot_single.png"

   plt.gcf().savefig(path)

   result = compare_images(baseline, path, tol=0.01)
   assert (result == None)
'''
   
# def test_classification_plot_interactions_w_baseline():
#     X, y = load_breast_cancer(return_X_y=True, as_frame=True)

#     scaler = StandardScaler()

#     X_names = X.columns

#     X = scaler.fit_transform(X)

#     X = pd.DataFrame(X, columns=X_names)

#     model = igann.IGANN(interactions=2, random_state=42)

#     model.fit(X, y)

#     model.plot_interactions()

#     baseline = "baseline/baseline_class_plot_interactions.png"

#     path = "temp_class_plot_interactions.png"

#     plt.gcf().savefig(path)

#     result = compare_images(baseline, path, tol=0.01)
#     assert (result == None)

# def test_classification_plot_learning_w_baseline():
#     X, y = load_breast_cancer(return_X_y=True, as_frame=True)

#     scaler = StandardScaler()

#     X_names = X.columns

#     X = scaler.fit_transform(X)

#     X = pd.DataFrame(X, columns=X_names)

#     model = igann.IGANN(random_state=42) # interactions=2, 

#     model.fit(X, y)
    
#     model.plot_learning()

#     baseline = "baseline/baseline_class_plot_learning.png"

#     path = "temp_class_plot_learning.png"

#     plt.gcf().savefig(path)

#     result = compare_images(baseline, path, tol=0.01)
#     assert (result == None)

# def test_regression_plot_single_w_baseline():
#    X, y = load_diabetes(return_X_y=True, as_frame=True)

#    scaler = StandardScaler()

#    X_names = X.columns

#    X = scaler.fit_transform(X)
#    y = (y - y.mean()) / y.std()

#    X = pd.DataFrame(X, columns=X_names)

#    model = igann.IGANN(task='regression', random_state=42) # , interactions=0

#    model.fit(X, y)

#    model.plot_single()

#    baseline = "baseline/baseline_reg_plot_single.png"

#    path = "temp_reg_plot_single.png"

#    plt.gcf().savefig(path)

#    result = compare_images(baseline, path, tol=0.05)
#    assert (result == None)
    
# def test_regression_plot_interactions_w_baseline():
#     X, y = load_diabetes(return_X_y=True, as_frame=True)

#     scaler = StandardScaler()

#     X_names = X.columns

#     X = scaler.fit_transform(X)

#     X = pd.DataFrame(X, columns=X_names)

#     model = igann.IGANN(task='regression', interactions=2, random_state=42)

#     model.fit(X, y)

#     model.plot_interactions()

#     baseline = "baseline/baseline_reg_plot_interactions.png"

#     path = "temp_reg_plot_interactions.png"

#     plt.gcf().savefig(path)

#     result = compare_images(baseline, path, tol=0.03)
#     assert (result == None)


def test_cat_variables():
    X, y = make_regression(100, 10, n_informative=3, random_state=0)
    y = (y - y.mean()) / y.std()
    X = pd.DataFrame(X)
    X['cat_test'] = np.random.choice(['A', 'B', 'C', 'D'], X.shape[0], p=[0.2, 0.2, 0.1, 0.5])
    m = igann.IGANN(task='regression', n_estimators=1000)
    m.fit(pd.DataFrame(X), y)
    assert m.n_categorical_cols == 3
    assert m.n_numerical_cols == 10

def test_igann_dummies_for_cat_with_nans():
    X, y = make_regression(100, 10, n_informative=3, random_state=0)
    y = (y - y.mean()) / y.std()
    X = pd.DataFrame(X)
    X['cat_test'] = np.random.choice(['A', 'B', 'C', 'D', np.nan], X.shape[0], p=[0.2, 0.2, 0.1, 0.3, 0.2])
    m = igann.IGANN(task='regression', n_estimators=1000)
    m.fit(pd.DataFrame(X), y)
    assert m.n_categorical_cols == 4
    assert m.n_numerical_cols == 10
    assert len(m.feature_names) == 14
    X['cat_test'] = np.random.choice(['A', 'B', 'C', np.nan], X.shape[0], p=[0.2, 0.2, 0.1, 0.5])
    m = igann.IGANN(task='regression', n_estimators=1000)
    m.fit(pd.DataFrame(X), y)
    assert m.n_categorical_cols == 3
    assert m.n_numerical_cols == 10
    assert len(m.feature_names) == 13

def test_torch_ridge():
    device="cpu"
    X = torch.tensor([[0.4259, 0.6296, 0.7241, 0.1714, 0.6942],
                      [0.4327, 0.9518, 0.5472, 0.7580, 0.6477],
                      [0.9833, 0.9742, 0.4205, 0.7979, 0.9641],
                      [0.7738, 0.0795, 0.4846, 0.6665, 0.8383],
                      [0.2489, 0.8715, 0.3307, 0.0499, 0.5630]])
    y = torch.tensor([0.5545, 0.8965, 0.6915, 0.7395, 0.3370])
    ridge = igann.torch_Ridge(alpha=0.0001, device=device)
    ridge.fit(X,y)
    X_test = torch.tensor([0.2581, 0.6990, 0.3861, 0.6054, 0.8429])
    pred = ridge.predict(X_test)
    assert isinstance(pred.item(), float)
    assert (round(pred.item(), 4) == 1.0188)
    ridge = igann.torch_Ridge(alpha=0.001, device=device)
    ridge.fit(X,y)
    pred = ridge.predict(X_test)
    assert (round(pred.item(), 4) == 0.9917)

def test_elm():
    X = torch.tensor([[-1.3496,  0.1114,  0.0340, -0.4152,  0.0000,  0.0000,  0.0000,  0.0000,
            0.0000,  1.0000],
            [ 0.8577, -1.0025, -0.0190, -0.1599,  0.0000,  0.0000,  1.0000,  0.0000,
            0.0000,  1.0000],
            [ 0.2599, -1.3205, -1.2370,  0.7818,  1.0000,  0.0000,  0.0000,  1.0000,
            0.0000,  0.0000],
            [-1.7787, -0.0556,  0.6544,  1.4960,  1.0000,  0.0000,  0.0000,  0.0000,
            0.0000,  1.0000],
            [-1.4063,  0.7601, -1.5047, -0.0831,  0.0000,  0.0000,  1.0000,  0.0000,
            0.0000,  1.0000]])
    y = torch.tensor([-0.4383, -0.0506, -1.3534, -0.0438, -1.4075])
    elm = igann.ELM_Regressor(X.shape[1], 6, X.shape[1], seed=0, elm_scale=10, 
                 elm_alpha=0.0001, act='elu', device='cpu')
    elm.fit(X, y, torch.sqrt(torch.tensor(0.5) * 0.1 * 1))
    X_test = torch.tensor([[-1.3496,-1.0025, -1.3205, 0.7601, 1.0,  0.0,  0.0, 1.0, 0.0, 0.0]])
    pred = elm.predict(X_test)
    print(round(pred.item(), 1))
    assert (round(pred.item(), 1) == -7.8)
test_elm()
'''
def test_igann_bagged():
    X, y = make_regression(1000, 4, n_informative=4, random_state=42)
    X = pd.DataFrame(X)
    X['cat_test'] = np.random.choice(['A', 'B', 'C', 'D'], X.shape[0], p=[0.2, 0.2, 0.1, 0.5])
    X['cat_test_2'] = np.random.choice(['E', 'F', 'G', 'H'], X.shape[0], p=[0.2, 0.2, 0.1, 0.5])
    X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=42)
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std
    m = igann.IGANN_Bagged(task='regression', n_estimators=100, verbose=0, n_bags=5)
    m.fit(pd.DataFrame(X_train), y_train)
    m.plot_single(show_n=6, max_cat_plotted=4)
    pred = m.predict(X_test)
    pred_proba = m.predict_proba(X_test)
    assert isinstance(pred, tuple)
    assert isinstance(pred_proba, tuple)
    assert isinstance(pred[0], np.ndarray)
    assert isinstance(pred_proba[0], np.ndarray)
    assert isinstance(pred[1], np.ndarray)
    assert isinstance(pred_proba[1], np.ndarray)
    assert len(pred) == 2
    assert len(pred_proba) == 2
    assert pred[0].shape[0] == len(X_test)
    assert pred_proba[0].shape[0] == len(X_test)
    assert pred[1].shape[0] == len(X_test)
    assert pred_proba[1].shape[0] == len(X_test)
    assert len(m.bags) == 5
'''
def test_parameters_n_hid():
    X, y = make_regression(10000, 10, n_informative=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    model = igann.IGANN(task='regression')
    assert (model.n_hid == 10) # If this fails, maybe the default value has changed
    model.fit(pd.DataFrame(X_train), y_train)
    assert (model.regressors[0].n_hid == 10) # If this fails, maybe the default value has changed

    model = igann.IGANN(task='regression', n_hid=15)
    assert (model.n_hid == 15)
    model.fit(pd.DataFrame(X_train), y_train)
    assert (model.regressors[0].n_hid == 15)

    model = igann.IGANN(task='regression', n_hid=5)
    assert (model.n_hid == 5)
    model.fit(pd.DataFrame(X_train), y_train)
    assert (model.regressors[0].n_hid == 5)

def test_parameters_n_estimators():
    X, y = make_regression(10000, 10, n_informative=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    model = igann.IGANN(task='regression')
    assert (model.n_estimators == 5000) # If this fails, maybe the default value has changed
    model.fit(pd.DataFrame(X_train), y_train)
    assert (len(model.regressors) <= 5000)

    model = igann.IGANN(task='regression', n_estimators=10000)
    assert (model.n_estimators == 10000)
    model.fit(pd.DataFrame(X_train), y_train)
    assert (len(model.regressors) <= 10000)

    model = igann.IGANN(task='regression', n_estimators=200)
    assert (model.n_estimators == 200)
    model.fit(pd.DataFrame(X_train), y_train)
    assert (len(model.regressors) <= 200)

def test_parameters_boost_rate():
    model = igann.IGANN(task='regression')
    assert (model.boost_rate == .1) # If this fails, maybe the default value has changed

    model = igann.IGANN(task='regression', boost_rate=0.3)
    assert (model.boost_rate == .3)

    model = igann.IGANN(task='regression', boost_rate=0.01)
    assert (model.boost_rate == .01)

def test_parameters_init_reg():
    X, y = make_regression(10000, 10, n_informative=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    model = igann.IGANN(task='regression')
    assert (model.init_reg == 1) # If this fails, maybe the default value has changed

    model = igann.IGANN(task='regression', init_reg=3)
    assert (model.init_reg == 3)

    model = igann.IGANN(task='regression', init_reg=0.1)
    assert (model.init_reg == 0.1)

def test_parameters_elm_scale():
    X, y = make_regression(10000, 10, n_informative=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    model = igann.IGANN(task='regression')
    assert (model.elm_scale == 1) # If this fails, maybe the default value has changed
    model.fit(pd.DataFrame(X_train), y_train)
    assert (model.regressors[0].elm_scale == 1)

    model = igann.IGANN(task='regression', elm_scale=3)
    assert (model.elm_scale == 3)
    model.fit(pd.DataFrame(X_train), y_train)
    assert (model.regressors[0].elm_scale == 3)

    model = igann.IGANN(task='regression', elm_scale=0.1)
    assert (model.elm_scale == 0.1)
    model.fit(pd.DataFrame(X_train), y_train)
    assert (model.regressors[0].elm_scale == 0.1)

def test_parameters_elm_alpha():
    X, y = make_regression(10000, 10, n_informative=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    model = igann.IGANN(task='regression')
    assert (model.elm_alpha == 1) # If this fails, maybe the default value has changed
    model.fit(pd.DataFrame(X_train), y_train)
    assert (model.regressors[0].elm_alpha == 1)

    model = igann.IGANN(task='regression', elm_alpha=3)
    assert (model.elm_alpha == 3)
    model.fit(pd.DataFrame(X_train), y_train)
    assert (model.regressors[0].elm_alpha == 3)

    model = igann.IGANN(task='regression', elm_alpha=0.1)
    assert (model.elm_alpha == 0.1)
    model.fit(pd.DataFrame(X_train), y_train)
    assert (model.regressors[0].elm_alpha == 0.1)

def test_parameters_act():
    X, y = make_regression(10000, 10, n_informative=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    model = igann.IGANN(task='regression')
    assert (model.act == "elu") # If this fails, maybe the default value has changed
    model.fit(pd.DataFrame(X_train), y_train)
    assert isinstance(model.regressors[0].act, torch.nn.ELU)

    model = igann.IGANN(task='regression', act="relu")
    assert (model.act == "relu")
    model.fit(pd.DataFrame(X_train), y_train)
    assert isinstance(model.regressors[0].act, torch.nn.ReLU)

    model = igann.IGANN(task='regression', act=torch.nn.Tanh())
    assert isinstance(model.act, torch.nn.Tanh)
    model.fit(pd.DataFrame(X_train), y_train)
    assert isinstance(model.regressors[0].act, torch.nn.Tanh)

def test_parameters_early_stopping():
    X, y = make_regression(10000, 10, n_informative=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    model = igann.IGANN(task='regression')
    assert (model.early_stopping == 50) # If this fails, maybe the default value has changed

    model = igann.IGANN(task='regression', early_stopping=20)
    assert (model.early_stopping == 20)

    model = igann.IGANN(task='regression', early_stopping=100)
    assert (model.early_stopping == 100)
