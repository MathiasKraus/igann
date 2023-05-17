import igann
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images

from sklearn.datasets import load_breast_cancer, load_diabetes, make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, mean_squared_error

import numpy as np

import torch


def test_sparse_igann():
    X, y = make_regression(100000, 10, n_informative=3, random_state=0)
    y = (y - y.mean()) / y.std()
    m =  igann.IGANN(task='regression', n_estimators=1000, sparse=10)
    m.fit(pd.DataFrame(X), y)
    assert len(m.feature_names) < 7

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

def test_classification_plot_single_w_baseline():
   X, y = load_breast_cancer(return_X_y=True, as_frame=True)

   scaler = StandardScaler()

   X_names = X.columns

   X = scaler.fit_transform(X)

   X = pd.DataFrame(X, columns=X_names)

   model = igann.IGANN(random_state=42) # interactions=0, 

   model.fit(X, y)

   model.plot_single()

   baseline = "baseline/baseline_class_plot_single.png"

   path = "temp_class_plot_single.png"

   plt.gcf().savefig(path)

   result = compare_images(baseline, path, tol=0.01)
   assert (result == None)

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

def test_regression_plot_single_w_baseline():
   X, y = load_diabetes(return_X_y=True, as_frame=True)

   scaler = StandardScaler()

   X_names = X.columns

   X = scaler.fit_transform(X)
   y = (y - y.mean()) / y.std()

   X = pd.DataFrame(X, columns=X_names)

   model = igann.IGANN(task='regression', random_state=42) # , interactions=0

   model.fit(X, y)

   model.plot_single()

   baseline = "baseline/baseline_reg_plot_single.png"

   path = "temp_reg_plot_single.png"

   plt.gcf().savefig(path)

   result = compare_images(baseline, path, tol=0.05)
   assert (result == None)
    
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
    assert (model.regressors[0].scale == 1)

    model = igann.IGANN(task='regression', elm_scale=3)
    assert (model.elm_scale == 3)
    model.fit(pd.DataFrame(X_train), y_train)
    assert (model.regressors[0].scale == 3)

    model = igann.IGANN(task='regression', elm_scale=0.1)
    assert (model.elm_scale == 0.1)
    model.fit(pd.DataFrame(X_train), y_train)
    assert (model.regressors[0].scale == 0.1)

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
