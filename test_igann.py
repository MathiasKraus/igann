import igann
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images

from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_squared_error

import numpy as np

def test_classification_train_no_interaction_pd_df():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    scaler = StandardScaler()

    X_names = X.columns

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train, columns=X_names)
    X_test = pd.DataFrame(X_test, columns=X_names)

    model = igann.IGANN()

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    f1 = f1_score(y_test, preds)

    assert (f1 > 0.94)
    # f1 above 0.98 in prior tests
'''
def test_classification_train_find_interactions_pd_df():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    scaler = StandardScaler()

    X_names = X.columns

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train, columns=X_names)
    X_test = pd.DataFrame(X_test, columns=X_names)

    model = igann.IGANN(interactions=2)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    f1 = f1_score(y_test, preds)

    assert (f1 > 0.94)
    # f1 above 0.98 in prior tests
'''
def test_classification_train_no_interaction_np_array():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    scaler = StandardScaler()

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = igann.IGANN()

    model.fit(X_train, y_train.to_numpy())
    preds = model.predict(X_test)

    f1 = f1_score(y_test.to_numpy(), preds)

    assert (f1 > 0.94)
    # f1 above 0.98 in prior tests
'''
def test_classification_train_find_interactions_np_array():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    scaler = StandardScaler()

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = igann.IGANN(interactions=2)

    model.fit(X_train, y_train.to_numpy())
    preds = model.predict(X_test)

    f1 = f1_score(y_test.to_numpy(), preds)

    assert (f1 > 0.94)
    # f1 above 0.98 in prior tests
'''    
def test_classification_predict_proba_no_interaction_np_array():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    scaler = StandardScaler()

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = igann.IGANN()

    model.fit(X_train, y_train.to_numpy())

    preds = model.predict_proba(X_test)

    max_indices = np.argmax(preds, axis=1)
    f1 = f1_score(y_test.to_numpy(), max_indices)

    assert (f1 > 0.94)
    # f1 above 0.97 in prior tests

def test_classification_predict_proba_no_interaction_pd_df():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    scaler = StandardScaler()

    X_names = X.columns

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train, columns=X_names)
    X_test = pd.DataFrame(X_test, columns=X_names)

    model = igann.IGANN()

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

    model = igann.IGANN()

    model.fit(X, y)

    model.plot_single()
'''
def test_classification_plot_interactions():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    scaler = StandardScaler()

    X_names = X.columns

    X = scaler.fit_transform(X)

    X = pd.DataFrame(X, columns=X_names)

    model = igann.IGANN(interactions=2)

    model.fit(X, y)

    model.plot_interactions()
'''
def test_classification_plot_learning():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    scaler = StandardScaler()

    X_names = X.columns

    X = scaler.fit_transform(X)

    X = pd.DataFrame(X, columns=X_names)

    model = igann.IGANN()

    model.fit(X, y)
    
    model.plot_learning()

def test_regression_train_no_interaction_pd_df():
    X, y = load_diabetes(return_X_y=True, as_frame=True)

    scaler = StandardScaler()

    X_names = X.columns

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train, columns=X_names)
    X_test = pd.DataFrame(X_test, columns=X_names)

    model = igann.IGANN(task='regression')

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)

    assert (mse < 3300)
    # mse below 3100 in prior tests
'''
def test_regression_train_find_interactions_pd_df():
    X, y = load_diabetes(return_X_y=True, as_frame=True)

    scaler = StandardScaler()

    X_names = X.columns

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train, columns=X_names)
    X_test = pd.DataFrame(X_test, columns=X_names)

    model = igann.IGANN(task='regression', interactions=2)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)

    assert (mse < 3300)
    # mse below 3100 in prior tests
'''

def test_regression_train_no_interaction_np_array():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    scaler = StandardScaler()

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = igann.IGANN(task='regression')

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)

    assert (mse < 3300)
    # mse below 3100 in prior tests
'''
def test_regression_train_find_interactions_np_array():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    scaler = StandardScaler()

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = igann.IGANN(task='regression', interactions=2)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)

    assert (mse < 3300)
    # mse below 3100 in prior tests
'''
def test_regression_plot_single():
    X, y = load_diabetes(return_X_y=True, as_frame=True)

    scaler = StandardScaler()

    X_names = X.columns

    X = scaler.fit_transform(X)

    X = pd.DataFrame(X, columns=X_names)

    model = igann.IGANN(task='regression')

    model.fit(X, y)

    model.plot_single()
    
'''
def test_regression_plot_interactions():
    X, y = load_diabetes(return_X_y=True, as_frame=True)

    scaler = StandardScaler()

    X_names = X.columns

    X = scaler.fit_transform(X)

    X = pd.DataFrame(X, columns=X_names)

    model = igann.IGANN(task='regression', interactions=2)

    model.fit(X, y)

    model.plot_interactions()
'''
def test_regression_plot_learning():
    X, y = load_diabetes(return_X_y=True, as_frame=True)

    scaler = StandardScaler()

    X_names = X.columns

    X = scaler.fit_transform(X)

    X = pd.DataFrame(X, columns=X_names)

    model = igann.IGANN(task='regression')

    model.fit(X, y)

    model.plot_learning()
'''
def test_classification_plot_single_w_baseline():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    scaler = StandardScaler()

    X_names = X.columns

    X = scaler.fit_transform(X)

    X = pd.DataFrame(X, columns=X_names)

    model = igann.IGANN(random_state=42)

    model.fit(X, y)

    model.plot_single()

    baseline = "baseline/baseline_class_plot_single.png"

    path = "temp_class_plot_single.png"

    plt.gcf().savefig(path)

    result = compare_images(baseline, path, tol=0.01)
    assert (result == None)

def test_classification_plot_interactions_w_baseline():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    scaler = StandardScaler()

    X_names = X.columns

    X = scaler.fit_transform(X)

    X = pd.DataFrame(X, columns=X_names)

    model = igann.IGANN(interactions=2, random_state=42)

    model.fit(X, y)

    model.plot_interactions()

    baseline = "baseline/baseline_class_plot_interactions.png"

    path = "temp_class_plot_interactions.png"

    plt.gcf().savefig(path)

    result = compare_images(baseline, path, tol=0.01)
    assert (result == None)

def test_classification_plot_learning_w_baseline():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    scaler = StandardScaler()

    X_names = X.columns

    X = scaler.fit_transform(X)

    X = pd.DataFrame(X, columns=X_names)

    model = igann.IGANN(interactions=2, random_state=42)

    model.fit(X, y)
    
    model.plot_learning()

    baseline = "baseline/baseline_class_plot_learning.png"

    path = "temp_class_plot_learning.png"

    plt.gcf().savefig(path)

    result = compare_images(baseline, path, tol=0.01)
    assert (result == None)

def test_regression_plot_single_w_baseline():
    X, y = load_diabetes(return_X_y=True, as_frame=True)

    scaler = StandardScaler()

    X_names = X.columns

    X = scaler.fit_transform(X)

    X = pd.DataFrame(X, columns=X_names)

    model = igann.IGANN(task='regression', interactions=0, random_state=42)

    model.fit(X, y)

    model.plot_single()

    baseline = "baseline/baseline_reg_plot_single.png"

    path = "temp_reg_plot_single.png"

    plt.gcf().savefig(path)

    result = compare_images(baseline, path, tol=0.03)
    assert (result == None)
    

def test_regression_plot_interactions_w_baseline():
    X, y = load_diabetes(return_X_y=True, as_frame=True)

    scaler = StandardScaler()

    X_names = X.columns

    X = scaler.fit_transform(X)

    X = pd.DataFrame(X, columns=X_names)

    model = igann.IGANN(task='regression', interactions=2, random_state=42)

    model.fit(X, y)

    model.plot_interactions()

    baseline = "baseline/baseline_reg_plot_interactions.png"

    path = "temp_reg_plot_interactions.png"

    plt.gcf().savefig(path)

    result = compare_images(baseline, path, tol=0.03)
    assert (result == None)
'''
