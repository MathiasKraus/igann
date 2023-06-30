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
- sparse: Tells if IGANN should be sparse or not. Integer denotes the max number of used features
- act: the activation function in the ELM model. Can be 'elu', 'relu' or a torch activation function.
- early_stopping: we use early stopping which means that we don't continue training more ELM. This parameter sets the patience
- models, if there has been no improvements for 'early_stopping' number of iterations.
- device: the device on which the model is optimized. Can be 'cpu' or 'cuda'
- random_state: random seed.
- optimize_threshold: if True, the threshold for the classification is optimized using train data only and using the ROC curve. Otherwise, per default the raw logit value greater 0 means class 1 and less 0 means class -1.
- verbose: verbosity level. Can be 0 for no information, 1 for printing losses, and 2 for plotting shape functions every 5 iterations.

## Examples
### Basic regression example

In the following, we use the common diabetes dataset from sklearn (https://scikit-learn.org/0.16/modules/generated/sklearn.datasets.load_diabetes.html). After loading the dataset via


```
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
model.plot_single(plot_by_list=['age', 'bmi', 'bp', 'sex', 's1', 's2'])
```
we obtain the following shape functions

![image](https://github.com/MathiasKraus/igann/assets/15181429/9c0607a9-f4ac-4515-b098-22500aef147b)


### Sparse regression example

In many cases, it makes sense to train a sparse IGANN model, i.e., a model which only basis its output on few features. This generally increases the ease of understanding the model behavior.

```
from igann import IGANN
model = IGANN(task='regression', sparse=5)
model.fit(X, y)
model.plot_single()
```

yields (note that the sparse parameters denotes the max number of features)

![image](https://github.com/MathiasKraus/igann/assets/15181429/1ef6a099-4e09-471a-9e6f-da955dbff23d)

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




