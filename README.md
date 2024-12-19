# IGANN - Interpretable Generalized Additive Neural Networks

--- This project is under active development ---

IGANN is a novel machine learning model from the family of generalized additive models (GAMs). This GAM is special in the sense that it uses a combination of extreme learning machines and gradient boosting.

Some of the main features are:
- Training is very fast and can be performed on CPU and GPU  
- The rate of change of so-called shape functions can be influenced through hyperparameter ELM_scale  
- The initial model is simply linear, thus IGANN generally also performs well on small datasets  

Main developers of this project are:

Mathias Kraus, FAU Erlangen-Nürnberg  
Daniel Tschernutter, ETH Zurich  
Sven Weinzierl, FAU Erlangen-Nürnberg  
Patrick Zschech, FAU Erlangen-Nürnberg  
Nico Hambauer, FAU Erlangen-Nürnberg  
Sven Kruschel, FAU Erlangen-Nürnberg  
Lasse Bohlen, FAU Erlangen-Nürnberg  
Julian Rosenberger, FAU Erlangen-Nürnberg  
Theodor Stöcker, FAU Erlangen-Nürnberg

## Installation

Available through 
```
pip install igann
```
For the latest features, use this repository.

## Dependencies

The project depends on PyTorch and abess (version 0.4.5).

## Usage

IGANN can in general be used similar to sklearn models. The methods to interact with IGANN are the following:
- .fit(X, y) for training IGANN on (X, y) dataset. Categorical variables are derived based on their dtypes. That's why the feature marix X needs to be a pandas DataFrame.
- .predict_raw(X) to compute simple prediction for regression or raw logits for classification tasks. Per default values greater than 0 could be interpreted as belonging to class 1 and values smaller than 0 as belonging to class -1. 
- .predict(X) to compute simple prediction for regression or class prediction in {-1, 1} for classification tasks. If the IGANN parameter optimize_threshold is set to True, the threshold for the class prediction is optimized on the training data and hence deviates from the decision boundary of 0.
- .predict_proba(X) to compute probability estimates
- .plot_single() to show shape functions
- .plot_learning() to show the learning curve on train and validation set


## Parameters

When initializing IGANN, the following parameters can be set:
- task: defines the task, can be 'regression' or 'classification'
- n_hid: the number of hidden neurons for one feature
- n_estimators: the maximum number of estimators (ELMs) to be fitted.
- boost_rate: Boosting rate.
- init_reg: the initial regularization strength for the linear model.
- elm_scale: the scale of the random weights in the elm model.
- elm_alpha: the regularization strength for the ridge regression in the ELM model.
- act: the activation function in the ELM model. Can be 'elu', 'relu' or a torch activation function.
- early_stopping: If there has been no improvements for 'early_stopping' number of iterations, training is stopped.
- device: the device on which the model is optimized. Can be 'cpu' or 'cuda'
- random_state: random seed.
- optimize_threshold: if True, the threshold for the classification is optimized using train data only and using the ROC curve. Otherwise, per default the raw logit value greater 0 means class 1 and less 0 means class -1.
- verbose: verbosity level. Can be 0 for no information, 1 for printing losses, and 2 for plotting shape functions every 5 iterations.

## Examples
### Basic regression example

In the following, we use the common diabetes dataset from sklearn (https://scikit-learn.org/0.16/modules/generated/sklearn.datasets.load_diabetes.html). After loading the dataset via


```
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

X, y = load_diabetes(return_X_y=True, as_frame=True)
scaler = StandardScaler()
X_names = X.columns

X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=X_names)
X['sex'] = X.sex.apply(lambda x: 'w' if x > 0 else 'm')
```
and !important scale the target values

```
y = (y - y.mean()) / y.std()
```

we can simply initialize and fit IGANN with
```
from igann import IGANN
model = IGANN(task='regression')
model.fit(X, y)
```

With 
```
model.plot_single()
```
we obtain the following shape functions

![image](https://github.com/MathiasKraus/igann/assets/15181429/9c0607a9-f4ac-4515-b098-22500aef147b)

### Scikit-Learn Integration with IGANN

or scikit-learn users, we offer IGANNClassifier and IGANNRegressor classes. These are optimized for scikit-learn's ecosystem, ensuring full compatibility with its tools and conventions. IGANNClassifier is ideal for classification tasks, and IGANNRegressor for regression. Both integrate smoothly with scikit-learn's features like cross-validation and grid search, allowing easy incorporation of IGANN's capabilities into your machine learning projects.

Import them directly from the igann package:
```
import pandas as pd
from igann import IGANNClassifier, IGANNRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_breast_cancer

# Load sample data
X, y = load_breast_cancer(return_X_y=True)

X = pd.DataFrame(X)

# Initialize the IGANNClassifier
igann_classifier = IGANNClassifier()

# Define the parameter grid to search
param_grid = {
    'boost_rate': [0.01, 0.1, 0.2],
    # add other parameters you wish to tune
}

# Create GridSearchCV object
grid_search = GridSearchCV(igann_classifier, param_grid, cv=5, scoring='accuracy')

# Perform grid search on the data
grid_search.fit(X, y)

# Best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
```

# Citations
```latex
@article{kraus2023interpretable,
  title={Interpretable Generalized Additive Neural Networks},
  author={Kraus, Mathias and Tschernutter, Daniel and Weinzierl, Sven and Zschech, Patrick},
  journal={European Journal of Operational Research},
  year={2023},
  publisher={Elsevier}
}
```




