import time
import torch
import warnings
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import abess.linear
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from pprint import pprint as pp

warnings.simplefilter("once", UserWarning)


import psutil
import time


class torch_Ridge:
    def __init__(self, alpha, device):
        self.coef_ = None
        self.alpha = alpha
        self.device = device

    def fit(self, X, y):
        self.coef_ = torch.linalg.solve(
            X.T @ X + self.alpha * torch.eye(X.shape[1]).to(self.device), X.T @ y
        )

    def predict(self, X):
        return X.to(self.device) @ self.coef_


class ELM_Regressor:
    """
    This class represents one single hidden layer neural network for a regression task.
    Trainable parameters are only the parameters from the output layer. The parameters
    of the hidden layer are sampled from a normal distribution. This increases the training
    performance significantly as it reduces the training task to a regularized linear
    regression (Ridge Regression), see "Extreme Learning Machines" for more details.
    """

    def __init__(
        self,
        n_input,
        n_categorical_cols,
        n_hid,
        seed=0,
        elm_scale=10,
        elm_alpha=0.0001,
        act="elu",
        device="cpu",
    ):
        """
        Input parameters:
        - n_input: number of inputs/features (should be X.shape[1])
        - n_cat_cols: number of categorically encoded features
        - n_hid: number of hidden neurons for the base functions
        - seed: This number sets the seed for generating the random weights. It should
                be different for each regressor
        - elm_scale: the scale which is used to initialize the weights in the hidden layer of the
                 model. These weights are not changed throughout the optimization.
        - elm_alpha: the regularization of the ridge regression.
        - act: the activation function in the model. can be 'elu', 'relu' or a torch activation function.
        - device: the device on which the regressor should train. can be 'cpu' or 'cuda'.
        """
        super().__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.n_numerical_cols = n_input - n_categorical_cols
        self.n_categorical_cols = n_categorical_cols
        # The following are the random weights in the model which are not optimized.
        self.hidden_list = torch.normal(
            mean=torch.zeros(self.n_numerical_cols, self.n_numerical_cols * n_hid),
            std=elm_scale,
        ).to(device)

        mask = torch.block_diag(*[torch.ones(n_hid)] * self.n_numerical_cols).to(device)
        self.hidden_mat = self.hidden_list * mask
        self.output_model = None
        self.n_input = n_input

        self.n_hid = n_hid
        self.elm_scale = elm_scale
        self.elm_alpha = elm_alpha
        if act == "elu":
            self.act = torch.nn.ELU()
        elif act == "relu":
            self.act = torch.nn.ReLU()
        else:
            self.act = act
        self.device = device

    def get_hidden_values(self, X):
        """
        This step computes the values in the hidden layer. For this, we iterate
        through the input features and multiply the feature values with the weights
        from hidden_list. After applying the activation function, we return the result
        in X_hid
        """
        X_hid = X[:, : self.n_numerical_cols] @ self.hidden_mat
        X_hid = self.act(X_hid)
        X_hid = torch.hstack((X_hid, X[:, self.n_numerical_cols :]))

        return X_hid

    def predict(self, X, hidden=False):
        """
        This function makes a full prediction with the model for a given input X.
        """
        if hidden:
            X_hid = X
        else:
            X_hid = self.get_hidden_values(X)

        # Now, we can use the values in the hidden layer to make the prediction with
        # our ridge regression
        out = X_hid @ self.output_model.coef_
        return out

    def predict_single(self, x, i):
        """
        This function computes the partial output of one base function for one feature.
        Note, that the bias term is not used for this prediction.
        Input parameters:
        x: a vector representing the values which are used for feature i
        i: the index of the feature that should be used for the prediction
        """

        # See self.predict for the description - it's almost equivalent.
        x_in = x.reshape(len(x), 1)
        if i < self.n_numerical_cols:
            # numerical feature
            x_in = x_in @ self.hidden_mat[
                i, i * self.n_hid : (i + 1) * self.n_hid
            ].unsqueeze(0)
            x_in = self.act(x_in)
            out = x_in @ self.output_model.coef_[
                i * self.n_hid : (i + 1) * self.n_hid
            ].unsqueeze(1)
        else:
            # categorical feature
            start_idx = self.n_numerical_cols * self.n_hid + (i - self.n_numerical_cols)
            out = x_in @ self.output_model.coef_[start_idx : start_idx + 1].unsqueeze(1)
        return out

    def fit(self, X, y, mult_coef):
        """
        This function fits the ELM on the training data (X, y).
        """
        X_hid = self.get_hidden_values(X)
        X_hid_mult = X_hid * mult_coef
        # Fit the ridge regression on the hidden values.
        m = torch_Ridge(alpha=self.elm_alpha, device=self.device)
        m.fit(X_hid_mult, y)
        self.output_model = m
        return X_hid


class IGANN:
    """
    This class represents the IGANN model. It can be used like a
    sklearn model (i.e., it includes .fit, .predict, .predict_proba, ...).
    The model can be used for a regression task or a binary classification task.
    For binary classification, the labels must be set to -1 and 1 (Note that labels with
    0 and 1 are transformed automatically). The model first fits a linear model and then
    subsequently adds ELMs according to a boosting framework.
    """

    def __init__(
        self,
        task="classification",
        n_hid=10,
        n_estimators=5000,
        boost_rate=0.1,
        init_reg=1,
        elm_scale=1,
        elm_alpha=1,
        sparse=0,
        act="elu",
        early_stopping=50,
        device="cpu",
        random_state=1,
        optimize_threshold=False,
        verbose=0,
        GAMwrapper=False,
        GAM_detail=100,
        regressor_limit=5001,
    ):
        """
        Initializes the model. Input parameters:
        task: defines the task, can be 'regression' or 'classification'
        n_hid: the number of hidden neurons for one feature
        n_estimators: the maximum number of estimators (ELMs) to be fitted.
        boost_rate: Boosting rate.
        init_reg: the initial regularization strength for the linear model.
        elm_scale: the scale of the random weights in the elm model.
        elm_alpha: the regularization strength for the ridge regression in the ELM model.
        sparse: Tells if IGANN should be sparse or not. Integer denotes the max number of used features
        act: the activation function in the ELM model. Can be 'elu', 'relu' or a torch activation function.
        early_stopping: we use early stopping which means that we don't continue training more ELM
        models, if there has been no improvements for 'early_stopping' number of iterations.
        device: the device on which the model is optimized. Can be 'cpu' or 'cuda'
        random_state: random seed.
        optimize_threshold: if True, the threshold for the classification is optimized using train data only and using the ROC curve. Otherwise, per default the raw logit value greater 0 means class 1 and less 0 means class -1.
        verbose: tells how much information should be printed when fitting. Can be 0 for (almost) no
        information, 1 for printing losses, and 2 for plotting shape functions in each iteration.
        """
        self.task = task
        self.n_hid = n_hid
        self.elm_scale = elm_scale
        self.elm_alpha = elm_alpha
        self.init_reg = init_reg
        self.act = act
        self.n_estimators = n_estimators
        self.early_stopping = early_stopping
        self.sparse = sparse
        self.device = device
        self.random_state = random_state
        self.optimize_threshold = optimize_threshold
        self.verbose = verbose

        self.boost_rate = boost_rate
        self.target_remapped_flag = False
        self.GAM = None
        self.GAMwrapper = GAMwrapper
        self.regressor_limit = regressor_limit
        self.GAM_detail = GAM_detail
        """Is set to true during the fit method if the target (y) is remapped to -1 and 1 instead of 0 and 1."""

    def _clip_p(self, p):
        if torch.max(p) > 100 or torch.min(p) < -100:
            warnings.warn(
                "Cutting prediction to [-100, 100]. Did you forget to scale y? Consider higher regularization elm_alpha."
            )
            return torch.clip(p, -100, 100)
        else:
            return p

    def _clip_p_numpy(self, p):
        if np.max(p) > 100 or np.min(p) < -100:
            warnings.warn(
                "Cutting prediction to [-100, 100]. Did you forget to scale y? Consider higher regularization elm_alpha."
            )
            return np.clip(p, -100, 100)
        else:
            return p

    def _loss_sqrt_hessian(self, y, p):
        """
        This function computes the square root of the hessians of the log loss or the mean squared error.
        """
        if self.task == "classification":
            return 0.5 / torch.cosh(0.5 * y * p)
        else:
            return torch.sqrt(torch.tensor([2.0]).to(self.device))

    def _get_y_tilde(self, y, p):
        if self.task == "classification":
            return y / torch.exp(0.5 * y * p)
        else:
            return torch.sqrt(torch.tensor(2.0).to(self.device)) * (y - p)

    def _reset_state(self):
        self.regressors = []
        self.boosting_rates = []
        self.train_scores = []
        self.val_scores = []
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.regressor_predictions = []

    def _preprocess_feature_matrix(self, X, fit_transform=True, return_tensor=True):
        """
        Preprocesses the feature matrix using ColumnTransformer for numerical scaling and one-hot encoding for categorical variables.
        Ensures numerical columns come first in the transformed matrix.
        """
        # Validate input
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "Please provide a pandas DataFrame as input for X. Processing stopped."
            )
            return

        # Standardize column names and sort them
        X.columns = [str(c) for c in X.columns]
        X = X.reindex(sorted(X.columns), axis=1)

        # Separate numerical and categorical columns we safe the column names for later reference
        self.categorical_cols = X.select_dtypes(
            include=["category", "object"]
        ).columns.tolist()
        self.numerical_cols = list(set(X.columns) - set(self.categorical_cols))

        # Define a ColumnTransformer
        if fit_transform:
            self.column_transformer = ColumnTransformer(
                transformers=[
                    # (
                    #     "num",
                    #     StandardScaler(),
                    #     self.numerical_cols,
                    # ),
                    (
                        "cat",
                        OneHotEncoder(
                            drop="first", handle_unknown="ignore", sparse_output=False
                        ),
                        self.categorical_cols,
                    ),
                ],
                remainder="passthrough",  # Keep other columns, if any
            )
            # Fit and transform the data
            X_transformed = self.column_transformer.fit_transform(X)
        else:
            # Transform using the pre-fitted ColumnTransformer
            X_transformed = self.column_transformer.transform(X)

        # Record feature names for reference
        self.feature_names = self.numerical_cols + list(
            self.column_transformer.named_transformers_["cat"].get_feature_names_out(
                self.categorical_cols
            )
        )

        # Identify dropped features from OneHotEncoder
        one_hot_encoder = self.column_transformer.named_transformers_["cat"]
        self.dropped_features = {
            feature: categories[0]
            for feature, categories in zip(
                self.categorical_cols, one_hot_encoder.categories_
            )
        }
        print(f"Dropped features: {self.dropped_features}")

        # Set the number of categorical and numerical columns based on OneHotEncoder output
        self.n_numerical_cols = len(self.numerical_cols)
        self.n_categorical_cols = len(
            self.column_transformer.named_transformers_["cat"].get_feature_names_out(
                self.categorical_cols
            )
        )

        if return_tensor:
            # Convert to PyTorch tensor
            X_tensor = torch.tensor(X_transformed, dtype=torch.float32)

        # Log details
        if self.verbose > 0:
            print(f"Feature names are set to: {self.feature_names}")
            print(f"Transformed shape: {X_tensor.shape}")

        return X_tensor

    def fit(
        self,
        X,
        y,
        val_set=None,
        eval=None,
        fitted_dummies=None,
    ):
        """
        This function fits the model on training data (X, y).
        Parameters:
        X: the feature matrix
        y: the targets
        val_set: can be tuple (X_val, y_val) for a defined validation set. If not set,
        it will be split from the training set randomly.
        eval: can be tuple (X_test, y_test) for additional evaluation during training
        This can be changed here to keep track of the same feature throughout training.
        """
        # Generate indices for splitting also straify ic classification
        # We do it with indices to be able to redo this with preprocessed data and regardless of the type
        indices = np.arange(len(X))
        train_indices, val_indices = train_test_split(
            indices,
            test_size=0.15,
            stratify=y if self.task == "classification" else None,
            random_state=self.random_state,
        )

        # Split the data into train and validation data (we will use this for predictions with the GAMwrapper)
        self.raw_X = X.copy()
        self.raw_X_train = X.iloc[train_indices]
        self.raw_X_val = X.iloc[val_indices]
        if type(y) == pd.Series or type(y) == pd.DataFrame:
            self.raw_y_train = y.iloc[train_indices]
            self.raw_y_val = y.iloc[val_indices]

        # just to make sure we use a fresh model and to be sklearn compatible
        self._reset_state()

        # Initialize the linear model based on the task
        if self.task == "classification":
            self.linear_model = LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=1 / self.init_reg,
                random_state=self.random_state,
            )
            self.criterion = lambda prediction, target: torch.nn.BCEWithLogitsLoss()(
                prediction, torch.nn.ReLU()(target)
            )
        elif self.task == "regression":
            self.linear_model = Lasso(alpha=self.init_reg)
            self.criterion = torch.nn.MSELoss()
        else:
            warnings.warn("Task not implemented. Can be classification or regression")

        # Preprocess the feature matrix including scaling and one-hot encoding
        X = self._preprocess_feature_matrix(X)

        # convert y to tensor
        if type(y) == pd.Series or type(y) == pd.DataFrame:
            y = y.values
        y = torch.from_numpy(y.squeeze()).float()

        # Remap targets for classification task
        if self.task == "classification":
            # In the case of targets in {0,1}, transform them to {-1,1} for optimization purposes
            if torch.min(y) != -1:
                self.target_remapped_flag = True
                y = 2 * y - 1

        # For whatever reason, we need that
        self.feature_indizes = np.arange(X.shape[1])

        # Fit the linear model on all data
        self.linear_model.fit(X, y)

        # Split the data into train and validation data (use indices to handle numpy or tensor)
        if val_set == None:
            X_train = X[train_indices]
            X_val = X[val_indices]
            y_train = y[train_indices]
            y_val = y[val_indices]

        else:
            X_train = X
            y_train = y
            X_val = val_set[0]
            y_val = val_set[1]

        # For Classification we work with the logits and not the probabilities. That's why we multiply X with
        # the coefficients and don't use the predict_proba function.
        if self.task == "classification":
            y_hat_train = torch.squeeze(
                torch.from_numpy(self.linear_model.coef_.astype(np.float32))
                @ torch.transpose(X_train, 0, 1)
            ) + float(self.linear_model.intercept_)
            y_hat_val = torch.squeeze(
                torch.from_numpy(self.linear_model.coef_.astype(np.float32))
                @ torch.transpose(X_val, 0, 1)
            ) + float(self.linear_model.intercept_)

        else:
            y_hat_train = torch.from_numpy(
                self.linear_model.predict(X_train).squeeze().astype(np.float32)
            )
            y_hat_val = torch.from_numpy(
                self.linear_model.predict(X_val).squeeze().astype(np.float32)
            )

        # Store some information about the dataset which we later use for plotting.
        # We still should decide if we want to use X, X_train for this.
        self.X_min = list(X.min(axis=0))
        self.X_max = list(X.max(axis=0))
        self.unique = [torch.unique(X[:, i]) for i in range(X.shape[1])]
        self.hist = [torch.histogram(X[:, i]) for i in range(X.shape[1])]

        if self.verbose >= 1:
            print("Training shape: {}".format(X.shape))
            print("Validation shape: {}".format(X_val.shape))
            print("Regularization: {}".format(self.init_reg))

        train_loss_init = self.criterion(y_hat_train, y_train)
        val_loss_init = self.criterion(y_hat_val, y_val)

        if self.verbose >= 1:
            print(
                "Train: {:.4f} Val: {:.4f} {}".format(
                    train_loss_init, val_loss_init, "init"
                )
            )

        X_train, y_train, y_hat_train, X_val, y_val, y_hat_val = (
            X_train.to(self.device),
            y_train.to(self.device),
            y_hat_train.to(self.device),
            X_val.to(self.device),
            y_val.to(self.device),
            y_hat_val.to(self.device),
        )

        self._run_optimization(
            X_train,
            y_train,
            y_hat_train,
            X_val,
            y_val,
            y_hat_val,
            eval,
            val_loss_init,
        )

        return

    def get_params(self, deep=True):
        return {
            "task": self.task,
            "n_hid": self.n_hid,
            "elm_scale": self.elm_scale,
            "elm_alpha": self.elm_alpha,
            "init_reg": self.init_reg,
            "act": self.act,
            "n_estimators": self.n_estimators,
            "early_stopping": self.early_stopping,
            "sparse": self.sparse,
            "device": self.device,
            "random_state": self.random_state,
            "optimize_threshold": self.optimize_threshold,
            "verbose": self.verbose,
            "boost_rate": self.boost_rate,
            # "target_remapped_flag": self.target_remapped_flag,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            if not hasattr(self, parameter):
                raise ValueError(
                    "Invalid parameter %s for estimator %s. Check the list of available parameters with `estimator.get_params().keys()`."
                    % (parameter, self)
                )
            setattr(self, parameter, value)
        return self

    def score(self, X, y, metric=None):
        predictions = self.predict(X)
        if self.task == "regression":
            # if these is no metric specified use default regression metric "mse"
            if metric is None:
                metric = "mse"
            metric_dict = {"mse": mean_squared_error, "r_2": r2_score}
            return metric_dict[metric](y, predictions)
        else:
            if metric is None:
                # if there is no metric specified use default classification metric "accuracy"
                metric = "accuracy"
            metric_dict = {
                "accuracy": accuracy_score,
                "precision": precision_score,
                "recall": recall_score,
                "f1": f1_score,
            }
            return metric_dict[metric](y, predictions)

    def _run_optimization(
        self,
        X,
        y,
        y_hat,
        X_val,
        y_val,
        y_hat_val,
        eval,
        best_loss,
    ):
        """
        This function runs the optimization for ELMs with single features. This function should not be called from outside.
        Parameters:
        X: the training feature matrix
        y: the training targets
        y_hat: the current prediction for y
        X_val: the validation feature matrix
        y_val: the validation targets
        y_hat_val: the current prediction for y_val
        eval: can be tuple (X_test, y_test) for additional evaluation during training
        best_loss: best previous loss achieved. This is to keep track of the overall best sequence of ELMs.
        This can be changed here to keep track of the same feature throughout training.
        """

        counter_no_progress = 0
        best_iter = 0

        # Sequentially fit one ELM after the other. Max number is stored in self.n_estimators.
        for counter in range(self.n_estimators):
            if len(self.regressors) > self.regressor_limit:
                print("Reached regressor limit compressing GAM")
                if self.GAMwrapper == True:
                    self.compress_to_GAM()
                    y_hat = torch.tensor(
                        self.predict_raw(self.raw_X_train), dtype=torch.float32
                    )
                    y_hat_val = torch.tensor(
                        self.predict_raw(self.raw_X_val), dtype=torch.float32
                    )
                    # print(f"y_hat is: {y_hat} and of type {y_hat.dtype}")
                    # print(f"y_hat_val is: {y_hat_val} and of type {y_hat_val.dtype}")

            hessian_train_sqrt = self._loss_sqrt_hessian(y, y_hat)
            y_tilde = torch.sqrt(torch.tensor(0.5).to(self.device)) * self._get_y_tilde(
                y, y_hat
            )

            # Init ELM
            regressor = ELM_Regressor(
                n_input=X.shape[1],
                n_categorical_cols=self.n_categorical_cols,
                n_hid=self.n_hid,
                seed=counter,
                elm_scale=self.elm_scale,
                elm_alpha=self.elm_alpha,
                act=self.act,
                device=self.device,
            )

            # Fit ELM regressor
            X_hid = regressor.fit(
                X,
                y_tilde,
                torch.sqrt(torch.tensor(0.5).to(self.device))
                * self.boost_rate
                * hessian_train_sqrt[:, None],
            )

            # Make a prediction of the ELM for the update of y_hat
            train_regressor_pred = regressor.predict(X_hid, hidden=True).squeeze()
            val_regressor_pred = regressor.predict(X_val).squeeze()

            self.regressor_predictions.append(train_regressor_pred)

            # Update the prediction for training and validation data
            # print(type(y_hat))
            y_hat += self.boost_rate * train_regressor_pred
            y_hat_val += self.boost_rate * val_regressor_pred

            y_hat = self._clip_p(y_hat)
            y_hat_val = self._clip_p(y_hat_val)

            train_loss = self.criterion(y_hat, y)
            val_loss = self.criterion(y_hat_val, y_val)

            # Keep the ELM, the boosting rate and losses in lists, so
            # we can later use them again.
            self.regressors.append(regressor)
            self.boosting_rates.append(self.boost_rate)
            self.train_losses.append(train_loss.cpu())
            self.val_losses.append(val_loss.cpu())

            # This is the early stopping mechanism. If there was no improvement on the
            # validation set, we increase a counter by 1. If there was an improvement,
            # we set it back to 0.
            counter_no_progress += 1
            if val_loss < best_loss:
                best_iter = counter + 1
                best_loss = val_loss
                counter_no_progress = 0

            if self.verbose >= 1:
                self._print_results(
                    counter,
                    counter_no_progress,
                    eval,
                    self.boost_rate,
                    train_loss,
                    val_loss,
                )

            # Stop training if the counter for early stopping is greater than the parameter we passed.
            if counter_no_progress > self.early_stopping and self.early_stopping > 0:
                break

            if self.verbose >= 2:
                if counter % 5 == 0:
                    if plot_fixed_features != None:
                        self.plot_single(plot_by_list=plot_fixed_features)
                    else:
                        self.plot_single()

        if self.early_stopping > 0:
            # We remove the ELMs that did not improve the performance. Most likely best_iter equals self.early_stopping.
            if self.verbose > 0:
                print(f"Cutting at {best_iter}")
            self.regressors = self.regressors[:best_iter]
            self.boosting_rates = self.boosting_rates[:best_iter]

        # if we use the GAMwrapper, we compress the ELMs to a GAM model in the end of the optimization
        if self.GAMwrapper == True:
            self.compress_to_GAM()
        return best_loss

    def _select_features(self, X, y):
        regressor = ELM_Regressor(
            X.shape[1],
            self.n_categorical_cols,
            self.n_hid,
            seed=0,
            elm_scale=self.elm_scale,
            act=self.act,
            device="cpu",
        )
        X_tilde = regressor.get_hidden_values(X)
        groups = self._flatten(
            [list(np.ones(self.n_hid) * i + 1) for i in range(self.n_numerical_cols)]
        )
        groups.extend(list(range(self.n_numerical_cols, X.shape[1])))

        if self.task == "classification":
            m = abess.linear.LogisticRegression(
                path_type="gs", cv=3, s_min=1, s_max=self.sparse, thread=0
            )
            m.fit(X_tilde.numpy(), np.where(y.numpy() == -1, 0, 1), group=groups)
        else:
            m = abess.linear.LinearRegression(
                path_type="gs", cv=3, s_min=1, s_max=self.sparse, thread=0
            )
            m.fit(X_tilde.numpy(), y, group=groups)

        active_num_features = np.where(
            np.sum(
                m.coef_[: self.n_numerical_cols * self.n_hid].reshape(-1, self.n_hid),
                axis=1,
            )
            != 0
        )[0]

        active_cat_features = (
            np.where(m.coef_[self.n_numerical_cols * self.n_hid :] != 0)[0]
            + self.n_numerical_cols
        )

        self.n_numerical_cols = len(active_num_features)
        self.n_categorical_cols = len(active_cat_features)

        active_features = list(active_num_features) + list(active_cat_features)

        if self.verbose > 0:
            print(f"Found features {active_features}")

        return active_features

    def _print_results(
        self, counter, counter_no_progress, eval, boost_rate, train_loss, val_loss
    ):
        """
        This function plots our results.
        """
        if counter_no_progress == 0:
            new_best_symb = "*"
        else:
            new_best_symb = ""
        if eval:
            test_pred = self.predict_raw(eval[0])
            test_loss = self.criterion(test_pred, eval[1])
            self.test_losses.append(test_loss)
            print(
                "{}{}: BoostRate: {:.3f}, Train loss: {:.5f} Val loss: {:.5f} Test loss: {:.5f}".format(
                    new_best_symb, counter, boost_rate, train_loss, val_loss, test_loss
                )
            )
        else:
            print(
                "{}{}: BoostRate: {:.3f}, Train loss: {:.5f} Val loss: {:.5f}".format(
                    new_best_symb, counter, boost_rate, train_loss, val_loss
                )
            )

    def _optimize_classification_threshold(self, X_train, y_train):
        """
        This function optimizes the classification threshold for the training set for later predictions.
        The use of the function is triggered by setting the parameter optimize_threshold to True.
        This is one method which does the job. However, we noticed that it is not always the best method and hence it
        defaults to no threshold optimization.
        """

        y_proba = self.predict_raw(X_train)

        # detach and numpy
        y_proba = y_proba.detach().cpu().numpy()
        y_train = y_train.detach().cpu().numpy()
        fpr, tpr, trs = roc_curve(y_train, y_proba)

        roc_scores = []
        thresholds = []
        for thres in trs:
            thresholds.append(thres)
            y_pred = np.where(y_proba > thres, 1, -1)
            # Apply desired utility function to y_preds, for example accuracy.
            roc_scores.append(roc_auc_score(y_train.squeeze(), y_pred.squeeze()))
        # convert roc_scores to numpy array
        roc_scores = np.array(roc_scores)
        # get the index of the best threshold
        ix = np.argmax(roc_scores)
        # get the best threshold
        return thresholds[ix]

    def predict_proba(self, X):
        """
        Similarly to sklearn, this function returns a matrix of the same length as X and two columns.
        The first column denotes the probability of class -1, and the second column denotes the
        probability of class 1.
        """
        if self.task == "regression":
            warnings.warn(
                "The call of predict_proba for a regression task was probably incorrect."
            )

        pred = self.predict_raw(X)
        pred = self._clip_p_numpy(pred)
        pred = 1 / (1 + np.exp(-pred))

        ret = np.zeros((len(X), 2), dtype=np.float32)
        ret[:, 1] = pred
        ret[:, 0] = 1 - pred

        return ret

    def predict(self, X):
        """
        This function returns a prediction for a given feature matrix X.
        Note: for a classification task, it returns the binary target values in a 1-d np.array, it can hold -1 and 1.
        If optimize_threshold is True for a classification task, the threshold is optimized on the training data.
        """
        if self.task == "regression":
            return self.predict_raw(X)
        else:
            pred_raw = self.predict_raw(X)
            # detach and numpy pred_raw
            if self.optimize_threshold:
                threshold = self.best_threshold
            else:
                threshold = 0
            pred = np.where(
                pred_raw < threshold,
                np.ones_like(pred_raw) * -1,
                np.ones_like(pred_raw),
            ).squeeze()

            if self.target_remapped_flag:
                pred = np.where(pred == -1, 0, 1)

            return pred

    def predict_raw(self, X):
        """
        This function returns a prediction for a given feature matrix X.
        Note: for a classification task, it returns the raw logit values.
        """
        # if we have a GAM wrapper, we use the GAM model for prediction
        if self.GAMwrapper == True and self.GAM is not None:
            # X = self._preprocess_feature_matrix(X, fit_dummies=False).to(self.device)

            # Its not really correct to name this nn, but since pred_nn i further processed this makes sensense
            pred_nn = self.GAM.predict_raw(X)
            # X = self._preprocess_feature_matrix(X, fit_dummies=False).to(self.device)
            # X = X[:, self.feature_indizes]
            pred = pred_nn + (self.linear_model.intercept_)
        else:
            X = self._preprocess_feature_matrix(X, fit_transform=False).to(self.device)
            X = X[:, self.feature_indizes]
            pred_nn = torch.zeros(len(X), dtype=torch.float32).to(self.device)
            for boost_rate, regressor in zip(self.boosting_rates, self.regressors):
                pred_nn += boost_rate * regressor.predict(X).squeeze()
            pred_nn = pred_nn.detach().cpu().numpy()
            X = X.detach().cpu()
            X = X.numpy()
            # print(f"here: X"type(X))
            pred = (
                pred_nn
                + (self.linear_model.coef_.astype(np.float32) @ X.transpose()).squeeze()
                + self.linear_model.intercept_
            )

        return pred

    def _flatten(self, l):
        return [item for sublist in l for item in sublist]

    def _split_long_titles(self, l):
        return "\n".join(l[p : p + 22] for p in range(0, len(l), 22))

    def _get_pred_of_i(self, i, x_values=None):
        if x_values == None:
            feat_values = self.unique[i]
        else:
            feat_values = x_values[i]
        # if there is a GAMwarapper and its feature dict is set up we use this for a prediction
        if self.GAMwrapper and self.GAM and self.GAM.feature_dict:
            pred = self.GAM.predict_single(i, feat_values)
            pred = torch.from_numpy(np.array(pred))
        else:
            if self.task == "classification":
                pred = self.linear_model.coef_[0, i] * feat_values
            else:
                pred = self.linear_model.coef_[i] * feat_values

        feat_values = feat_values.to(self.device)
        for regressor, boost_rate in zip(self.regressors, self.boosting_rates):
            pred += (
                boost_rate
                * regressor.predict_single(feat_values.reshape(-1, 1), i).squeeze()
            ).cpu()
        return feat_values, pred

    def get_shape_functions_as_dict(self, x_values=None):
        shape_functions = []
        # print(self.feature_names)
        for i, feat_name in enumerate(self.feature_names):
            datatype = "numerical" if i < self.n_numerical_cols else "categorical"
            feat_values, pred = self._get_pred_of_i(i, x_values)
            if datatype == "numerical":
                shape_functions.append(
                    {
                        "name": feat_name,
                        "datatype": datatype,
                        "x": feat_values.cpu().numpy(),
                        "y": pred.numpy(),
                        "avg_effect": float(torch.mean(torch.abs(pred))),
                        "hist": self.hist[i],
                    }
                )
            else:
                class_name = feat_name.split("_")[-1]
                shape_functions.append(
                    {
                        "name": feat_name.rsplit("_", 1)[0],
                        "datatype": datatype,
                        "x": [class_name],
                        "y": [pred.numpy()[1]],
                        "avg_effect": float(torch.mean(torch.abs(pred))),
                        "hist": [[self.hist[i][0][-1]], [class_name]],
                    }
                )
        # print(shape_functions)

        # todo: make this more efficient
        final_shape_functions = {}
        for shape_function in shape_functions:
            name = shape_function["name"]
            # if the feature is cateogrical we need to add the dropped class to the existing shape function
            if name in final_shape_functions.keys():
                final_shape_functions[name]["x"].extend(shape_function["x"])
                final_shape_functions[name]["y"].extend(shape_function["y"])
                final_shape_functions[name]["avg_effect"] += shape_function[
                    "avg_effect"
                ]
                final_shape_functions[name]["hist"][0].append(
                    shape_function["hist"][0][0]
                )
                final_shape_functions[name]["hist"][1].append(
                    shape_function["hist"][1][0]
                )
            # if the feature is numerical or categorical and not in the dict yet we just add it to the final shape functions
            else:
                final_shape_functions[name] = shape_function
        # pp(final_shape_functions)

        # if we have dropped features we need to add them to the final shape functions
        # todo:
        num_rows = self.raw_X.shape[0]
        for name, function in final_shape_functions.items():
            if final_shape_functions[name]["datatype"] == "categorical":
                class_name = str(self.dropped_features[name])
                final_shape_functions[name]["x"].append(class_name)
                final_shape_functions[name]["y"].append(0)  # droped class effect is 0

                final_shape_functions[name]["hist"][0].append(
                    torch.tensor(
                        num_rows - np.sum(final_shape_functions[name]["hist"][0])
                    )
                )
                final_shape_functions[name]["hist"][1].append(class_name)

        return final_shape_functions

    def plot_single(
        self, plot_by_list=None, show_n=5, scaler_dict=None, max_cat_plotted=4
    ):
        """
        This function plots the most important shape functions.
        Parameters:
        show_n: the number of shape functions that should be plotted.
        scaler_dict: dictionary that maps every numerical feature to the respective (sklearn) scaler.
                     scaler_dict[num_feature_name].inverse_transform(...) is called if scaler_dict is not None
        """
        shape_functions = self.get_shape_functions_as_dict()
        if plot_by_list is None:
            top_k = [
                d
                for d in sorted(
                    shape_functions, reverse=True, key=lambda x: x["avg_effect"]
                )
            ][:show_n]
            show_n = min(show_n, len(top_k))
        else:
            top_k = [
                d
                for d in sorted(
                    shape_functions, reverse=True, key=lambda x: x["avg_effect"]
                )
            ]
            show_n = len(plot_by_list)

        plt.close(fig="Shape functions")
        fig, axs = plt.subplots(
            2,
            show_n,
            figsize=(14, 4),
            gridspec_kw={"height_ratios": [5, 1]},
            num="Shape functions",
        )
        plt.subplots_adjust(wspace=0.4)

        i = 0
        for d in top_k:
            if plot_by_list is not None and d["name"] not in plot_by_list:
                continue
            if scaler_dict:
                d["x"] = (
                    scaler_dict[d["name"]]
                    .inverse_transform(d["x"].reshape(-1, 1))
                    .squeeze()
                )
            if d["datatype"] == "categorical":
                if show_n == 1:
                    d["y"] = np.array(d["y"])
                    d["x"] = np.array(d["x"])
                    hist_items = [d["hist"][0][0].item()]
                    hist_items.extend(his[0].item() for his in d["hist"][0][1:])

                    idxs_to_plot = np.argpartition(
                        np.abs(d["y"]),
                        (
                            -(len(d["y"]) - 1)
                            if len(d["y"]) <= (max_cat_plotted - 1)
                            else -(max_cat_plotted - 1)
                        ),
                    )[-(max_cat_plotted - 1) :]
                    y_to_plot = d["y"][idxs_to_plot]
                    x_to_plot = d["x"][idxs_to_plot].tolist()
                    hist_items_to_plot = [hist_items[i] for i in idxs_to_plot]
                    if len(d["x"]) > max_cat_plotted - 1:
                        # other classes:
                        if "others" in x_to_plot:
                            x_to_plot.append(
                                "others_" + str(np.random.randint(0, 999))
                            )  # others or else seem like plausible variable names
                        else:
                            x_to_plot.append("others")
                        y_to_plot = np.append(y_to_plot.flatten(), [[0]]).reshape(
                            max_cat_plotted,
                        )
                        hist_items_to_plot.append(
                            np.sum(
                                [
                                    hist_items[i]
                                    for i in range(len(hist_items))
                                    if i not in idxs_to_plot
                                ]
                            )
                        )

                    axs[0].bar(
                        x=x_to_plot, height=y_to_plot, width=0.5, color="darkblue"
                    )
                    axs[1].bar(
                        x=x_to_plot,
                        height=hist_items_to_plot,
                        width=1,
                        color="darkblue",
                    )

                    axs[0].set_title(
                        "{}:\n{:.2f}%".format(
                            self._split_long_titles(d["name"]), d["avg_effect"]
                        )
                    )
                    axs[0].grid()
                else:
                    d["y"] = np.array(d["y"])
                    d["x"] = np.array(d["x"])
                    hist_items = [d["hist"][0][0].item()]
                    hist_items.extend(his[0].item() for his in d["hist"][0][1:])

                    idxs_to_plot = np.argpartition(
                        np.abs(d["y"]),
                        (
                            -(len(d["y"]) - 1)
                            if len(d["y"]) <= (max_cat_plotted - 1)
                            else -(max_cat_plotted - 1)
                        ),
                    )[-(max_cat_plotted - 1) :]
                    y_to_plot = d["y"][idxs_to_plot]
                    x_to_plot = d["x"][idxs_to_plot].tolist()
                    hist_items_to_plot = [hist_items[i] for i in idxs_to_plot]
                    if len(d["x"]) > max_cat_plotted - 1:
                        # other classes:
                        if "others" in x_to_plot:
                            x_to_plot.append(
                                "others_" + str(np.random.randint(0, 999))
                            )  # others or else seem like plausible variable names
                        else:
                            x_to_plot.append("others")
                        y_to_plot = np.append(y_to_plot.flatten(), [[0]]).reshape(
                            max_cat_plotted,
                        )
                        hist_items_to_plot.append(
                            np.sum(
                                [
                                    hist_items[i]
                                    for i in range(len(hist_items))
                                    if i not in idxs_to_plot
                                ]
                            )
                        )

                    axs[0][i].bar(
                        x=x_to_plot, height=y_to_plot, width=0.5, color="darkblue"
                    )
                    axs[1][i].bar(
                        x=x_to_plot,
                        height=hist_items_to_plot,
                        width=1,
                        color="darkblue",
                    )

                    axs[0][i].set_title(
                        "{}:\n{:.2f}%".format(
                            self._split_long_titles(d["name"]), d["avg_effect"]
                        )
                    )
                    axs[0][i].grid()

            else:
                if show_n == 1:
                    g = sns.lineplot(
                        x=d["x"], y=d["y"], ax=axs[0], linewidth=2, color="darkblue"
                    )
                    g.axhline(y=0, color="grey", linestyle="--")
                    axs[1].bar(
                        d["hist"][1][:-1], d["hist"][0], width=1, color="darkblue"
                    )
                    axs[0].set_title(
                        "{}:\n{:.2f}%".format(
                            self._split_long_titles(d["name"]), d["avg_effect"]
                        )
                    )
                    axs[0].grid()
                else:
                    g = sns.lineplot(
                        x=d["x"], y=d["y"], ax=axs[0][i], linewidth=2, color="darkblue"
                    )
                    g.axhline(y=0, color="grey", linestyle="--")
                    axs[1][i].bar(
                        d["hist"][1][:-1], d["hist"][0], width=1, color="darkblue"
                    )
                    axs[0][i].set_title(
                        "{}:\n{:.2f}%".format(
                            self._split_long_titles(d["name"]), d["avg_effect"]
                        )
                    )
                    axs[0][i].grid()

            i += 1

        if show_n == 1:
            axs[1].get_xaxis().set_visible(False)
            axs[1].get_yaxis().set_visible(False)
        else:
            for i in range(show_n):
                axs[1][i].get_xaxis().set_visible(False)
                axs[1][i].get_yaxis().set_visible(False)
        plt.show()

    def plot_learning(self):
        """
        Plot the training and the validation losses over time (i.e., for the sequence of learned
        ELMs)
        """
        fig, axs = plt.subplots(1, 1, figsize=(16, 8))
        fig.axes[0].plot(
            np.arange(len(self.train_losses)), self.train_losses, label="Train"
        )
        fig.axes[0].plot(np.arange(len(self.val_losses)), self.val_losses, label="Val")
        if len(self.test_losses) > 0:
            fig.axes[0].plot(
                np.arange(len(self.test_losses)), self.test_losses, label="Test"
            )
        plt.legend()
        plt.show()

    def compress_to_GAM(self):
        """
        Compress the model to a GAM model. This is useful if the model is too large and the user wants to make fast predictions.
        """
        print("Compressing to GAM")
        if self.GAM is None:
            self.GAM = GAMmodel(self, self.task, self.GAM_detail)
        self.GAM.set_shape_functions()

        # ass we now use the shape function for the prediction we do not need the regressors or bossting rates.
        self.regressors = []
        self.boosting_rates = []


class GAMmodel:
    """
    The idea of this class is to create a simple model that copies the dicision logic of igann, but simple uses points and interpolation.
    This can be useful for applications where the model should be very small and fast in prediction.
    Further this could allow expert knowledge to be included in the model.
    """

    def __init__(
        self,
        model,
        task,
        detail=100,
    ):
        self.base_model = model
        self.task = task
        self.GAM = None
        self.feature_dict = {}
        self.detail = detail
        # print(self.base_model.feature_names)

    def get_feature_dict(self):
        return self.feature_dict

    def set_feature_dict(self, feat_dict):
        self.feature_dict = feat_dict
        # print("new feature_dict is set")
        return

    def update_feature_dict(self, feat_dict):
        self.feature_dict.update(feat_dict)
        return

    def set_shape_functions(self):
        shape_data = self.base_model.get_shape_functions_as_dict()
        # print(shape_data)
        for feature, feature_dict in shape_data.items():

            feature_name = feature_dict["name"]
            feature_type = feature_dict["datatype"]
            # print(feature_type)
            feature_x = feature_dict["x"]
            feature_y = feature_dict["y"]

            # for categorical features we need use one point per class
            if feature_type == "categorical":
                feature_x_new = feature_x
                feature_y_new = feature_y
            else:
                feature_x_new, feature_y_new = self.create_points(
                    feature_x, feature_y, self.detail
                )
            self.feature_dict[feature_name] = {
                "datatype": feature_type,
                "x": feature_x_new,
                "y": feature_y_new,
            }

    def create_points(self, X, Y, num_points):
        min_x, max_x = min(X), max(X)
        # print(min_x, max_x)
        x_values = np.linspace(min_x, max_x, num_points)
        artificial_points_X = []
        artificial_points_Y = []

        for x in x_values:
            # Find the indices of the points on either side of x
            idx1 = np.searchsorted(X, x)
            if idx1 == 0:
                y = Y[0]
            elif idx1 == len(X):
                y = Y[-1]
            else:
                x1, y1 = X[idx1 - 1], Y[idx1 - 1]
                x2, y2 = X[idx1], Y[idx1]
                # Compute the weighted average of the y-values of the points on either side
                y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
            # Append the artificial point
            artificial_points_X.append(x)
            artificial_points_Y.append(y)

        return artificial_points_X, artificial_points_Y

    def predict_single(self, feature_name, x):

        if type(feature_name) == int:
            feature_name = self.base_model.feature_names[feature_name]

        if feature_name not in self.feature_dict.keys():
            # print(f"Feature {feature_name} not in feature dict")
            new_feature_name = feature_name.rsplit("_", 1)[0]
            class_name = feature_name.rsplit("_", 1)[-1]
            # print(f"Feature name is now: {new_feature_name}")
            # print(f"Class name is now: {class_name}")
            # print(f"before: {x}")
            x = [
                (
                    class_name
                    if x == 1
                    else self.base_model.dropped_features[new_feature_name]
                )
                for x in x
            ]
            # print(f"after: {x}")
            feature_name = new_feature_name

        feature = self.feature_dict[feature_name]
        if feature["datatype"] == "categorical":
            x_classes = feature["x"]
            y_values = feature["y"]
            y = []
            for x in x:
                if str(x) in x_classes:
                    y.append(y_values[x_classes.index(str(x))])
                else:
                    y.append(0)

        else:
            x_values = feature["x"]
            y_values = feature["y"]
            y = np.interp(
                x, x_values, y_values
            )  # Linear interpolation # also strategies for interpolation beyond x limtis can be created here.
        # print(len(y))
        return y

    def predict_raw(self, X):
        """
        Predict raw values using scaled numerical features and original (raw) categorical features.
        """
        # Step 1: Access the numerical scaler
        num_scaler = self.base_model.column_transformer.named_transformers_["num"]

        # Step 2: Identify the numerical and categorical columns
        numerical_cols = self.base_model.numerical_cols  # Stored during preprocessing
        categorical_cols = (
            self.base_model.categorical_cols
        )  # Stored during preprocessing

        # Step 3: Scale the numerical columns
        X_num = num_scaler.transform(X[numerical_cols])

        # Convert scaled numerical data back to a DataFrame
        X_num = pd.DataFrame(X_num, columns=numerical_cols, index=X.index)

        # Step 4: Extract the original categorical columns (raw values)
        X_cat = X[categorical_cols]  # No scaling or transformation needed

        # Step 5: Combine scaled numerical and raw categorical features
        X_combined = pd.concat([X_num, X_cat], axis=1)

        # Step 6: Generate predictions using predict_single
        y = {}
        for col in X_combined.columns:
            y[col] = self.predict_single(col, X_combined[col])

        # Step 7: Aggregate the predictions
        y = pd.DataFrame(y)
        y_predict_raw = np.array(y.sum(axis=1))

        return y_predict_raw

    def predict_proba(self, df):
        y_predict_raw = self.predict_raw(df)
        y_predict_proba = 1 / (1 + np.exp(-y_predict_raw))
        return y_predict_proba

    def predict(self, df, threshold=0.5):
        y_predict_proba = self.predict_proba(df)
        y_predict = np.where(y_predict_proba > threshold, 1, 0)
        return y_predict


class IGANN_Bagged:
    def __init__(
        self,
        task="classification",
        n_hid=10,
        n_estimators=5000,
        boost_rate=0.1,
        init_reg=1,
        n_bags=3,
        elm_scale=1,
        elm_alpha=1,
        sparse=0,
        act="elu",
        early_stopping=50,
        device="cpu",
        random_state=1,
        optimize_threshold=False,
        verbose=0,
    ):
        2
        self.n_bags = n_bags
        self.sparse = sparse
        self.random_state = random_state
        self.bags = [
            IGANN(
                task,
                n_hid,
                n_estimators,
                boost_rate,
                init_reg,
                elm_scale,
                elm_alpha,
                sparse,
                act,
                early_stopping,
                device,
                random_state + i,
                optimize_threshold,
                verbose=verbose,
            )
            for i in range(n_bags)
        ]

    def fit(self, X, y, val_set=None, eval=None, plot_fixed_features=None):
        X.columns = [str(c) for c in X.columns]
        X = X.reindex(sorted(X.columns), axis=1)
        categorical_cols = sorted(
            X.select_dtypes(include=["category", "object"]).columns.tolist()
        )
        numerical_cols = sorted(list(set(X.columns) - set(categorical_cols)))

        if len(numerical_cols) > 0:
            X_num = torch.from_numpy(X[numerical_cols].values).float()
            self.n_numerical_cols = X_num.shape[1]
        else:
            self.n_numerical_cols = 0

        if len(categorical_cols) > 0:
            get_dummies = GetDummies(categorical_cols)
            get_dummies.fit(X[categorical_cols])
        else:
            get_dummies = None

        ctr = 0
        for b in self.bags:
            print("#")
            random_generator = np.random.Generator(
                np.random.PCG64(self.random_state + ctr)
            )
            ctr += 1
            idx = random_generator.choice(np.arange(len(X)), len(X))
            b.fit(
                X.iloc[idx], np.array(y)[idx], val_set, eval, fitted_dummies=get_dummies
            )

    def predict(self, X):
        preds = []
        for b in self.bags:
            preds.append(b.predict(X))
        return np.array(preds).mean(0), np.array(preds).std(0)

    def predict_proba(self, X):
        preds = []
        for b in self.bags:
            preds.append(b.predict_proba(X))
        return np.array(preds).mean(0), np.array(preds).std(0)

    def plot_single(
        self, plot_by_list=None, show_n=5, scaler_dict=None, max_cat_plotted=4
    ):
        x_values = dict()
        for i, feat_name in enumerate(self.bags[0].feature_names):
            curr_min = 2147483647
            curr_max = -2147483646
            most_unique = 0
            for b in self.bags:
                if b.X_min[0][i] < curr_min:
                    curr_min = b.X_min[0][i]
                if b.X_max[0][i] > curr_max:
                    curr_max = b.X_max[0][i]
                if len(b.unique[i]) > most_unique:
                    most_unique = len(b.unique[i])
            x_values[i] = torch.from_numpy(
                np.arange(curr_min, curr_max, 1.0 / most_unique)
            ).float()

        shape_functions = [
            b.get_shape_functions_as_dict(x_values=x_values) for b in self.bags
        ]

        avg_effects = {}
        for sf in shape_functions:
            for feat_d in sf:
                if feat_d["name"] in avg_effects:
                    avg_effects[feat_d["name"]].append(feat_d["avg_effect"])
                else:
                    avg_effects[feat_d["name"]] = [feat_d["avg_effect"]]

        for k, v in avg_effects.items():
            avg_effects[k] = np.mean(v)

        for sf in shape_functions:
            for feat_d in sf:
                feat_d["avg_effect"] = avg_effects[feat_d["name"]]

        if plot_by_list is None:
            top_k = [
                d
                for d in sorted(
                    shape_functions[0], reverse=True, key=lambda x: x["avg_effect"]
                )
            ][:show_n]
            show_n = min(show_n, len(top_k))
        else:
            top_k = [
                d
                for d in sorted(
                    shape_functions[0], reverse=True, key=lambda x: x["avg_effect"]
                )
            ]
            show_n = len(plot_by_list)

        plt.close(fig="Shape functions")
        fig, axs = plt.subplots(
            2,
            show_n,
            figsize=(14, 4),
            gridspec_kw={"height_ratios": [5, 1]},
            num="Shape functions",
        )
        plt.subplots_adjust(wspace=0.4)

        axs_i = 0
        for d in top_k:
            if plot_by_list is not None and d["name"] not in plot_by_list:
                continue
            if scaler_dict:
                for sf in shape_functions:
                    d["x"] = (
                        scaler_dict[d["name"]]
                        .inverse_transform(d["x"].reshape(-1, 1))
                        .squeeze()
                    )
            y_l = []
            for sf in shape_functions:
                for feat in sf:
                    if d["name"] == feat["name"]:
                        y_l.append(np.array(feat["y"]))
                    else:
                        continue
            y_mean = np.mean(y_l, axis=0)
            y_std = np.std(y_l, axis=0)
            y_mean_and_std = np.column_stack((y_mean, y_std))
            if show_n == 1:
                if d["datatype"] == "categorical":
                    hist_items = [d["hist"][0][0].item()]
                    hist_items.extend(his[0].item() for his in d["hist"][0][1:])

                    idxs_to_plot = np.argpartition(
                        np.abs(y_mean_and_std[:, 0]),
                        (
                            -(len(y_mean_and_std) - 1)
                            if len(y_mean_and_std) <= (max_cat_plotted - 1)
                            else -(max_cat_plotted - 1)
                        ),
                    )[-(max_cat_plotted - 1) :]
                    d_X = [d["x"][i] for i in idxs_to_plot]
                    y_mean_and_std_to_plot = y_mean_and_std[:, :][idxs_to_plot]
                    hist_items_to_plot = [hist_items[i] for i in idxs_to_plot]
                    if len(y_mean_and_std) > max_cat_plotted - 1:
                        # other classes:
                        if "others" in d_X:
                            d_X.append(
                                "others_" + str(np.random.randint(0, 999))
                            )  # others or else seem like plausible variable names
                        else:
                            d_X.append("others")
                        y_mean_and_std_to_plot = np.append(
                            y_mean_and_std_to_plot.flatten(), [[0, 0]]
                        ).reshape(max_cat_plotted, 2)
                        hist_items_to_plot.append(
                            np.sum(
                                [
                                    hist_items[i]
                                    for i in range(len(hist_items))
                                    if i not in idxs_to_plot
                                ]
                            )
                        )
                    axs[0].bar(
                        x=d_X,
                        height=y_mean_and_std_to_plot[:, 0],
                        width=0.5,
                        color="darkblue",
                    )
                    axs[0].errorbar(
                        x=d_X,
                        y=y_mean_and_std_to_plot[:, 0],
                        yerr=y_mean_and_std_to_plot[:, 1],
                        fmt="none",
                        color="black",
                        capsize=5,
                    )
                    axs[1].bar(
                        x=d_X, height=hist_items_to_plot, width=1, color="darkblue"
                    )
                else:
                    g = sns.lineplot(
                        x=d["x"],
                        y=y_mean_and_std[:, 0],
                        ax=axs[0],
                        linewidth=2,
                        color="darkblue",
                    )
                    g = axs[0].fill_between(
                        x=d["x"],
                        y1=y_mean_and_std[:, 0] - y_mean_and_std[:, 1],
                        y2=y_mean_and_std[:, 0] + y_mean_and_std[:, 1],
                        color="aqua",
                    )
                    axs[1].bar(
                        d["hist"][1][:-1], d["hist"][0], width=1, color="darkblue"
                    )
                axs[0].axhline(y=0, color="grey", linestyle="--")
                axs[0].set_title(
                    "{}:\n{:.2f}%".format(
                        self.bags[0]._split_long_titles(d["name"]), d["avg_effect"]
                    )
                )
                axs[0].grid()
            else:
                if d["datatype"] == "categorical":
                    hist_items = [d["hist"][0][0].item()]
                    hist_items.extend(his[0].item() for his in d["hist"][0][1:])

                    idxs_to_plot = np.argpartition(
                        np.abs(y_mean_and_std[:, 0]),
                        (
                            -(len(y_mean_and_std) - 1)
                            if len(y_mean_and_std) <= (max_cat_plotted - 1)
                            else -(max_cat_plotted - 1)
                        ),
                    )[-(max_cat_plotted - 1) :]
                    d_X = [d["x"][i] for i in idxs_to_plot]
                    y_mean_and_std_to_plot = y_mean_and_std[:, :][idxs_to_plot]
                    hist_items_to_plot = [hist_items[i] for i in idxs_to_plot]
                    if len(y_mean_and_std) > max_cat_plotted - 1:
                        # other classes:
                        if "others" in d_X:
                            d_X.append(
                                "others_" + str(np.random.randint(0, 999))
                            )  # others or else seem like plausible variable names
                        else:
                            d_X.append("others")
                        y_mean_and_std_to_plot = np.append(
                            y_mean_and_std_to_plot.flatten(), [[0, 0]]
                        ).reshape(max_cat_plotted, 2)
                        hist_items_to_plot.append(
                            np.sum(
                                [
                                    hist_items[i]
                                    for i in range(len(hist_items))
                                    if i not in idxs_to_plot
                                ]
                            )
                        )
                    axs[0][axs_i].bar(
                        x=d_X,
                        height=y_mean_and_std_to_plot[:, 0],
                        width=0.5,
                        color="darkblue",
                    )
                    axs[0][axs_i].errorbar(
                        x=d_X,
                        y=y_mean_and_std_to_plot[:, 0],
                        yerr=y_mean_and_std_to_plot[:, 1],
                        fmt="none",
                        color="black",
                        capsize=5,
                    )
                    axs[1][axs_i].bar(
                        x=d_X, height=hist_items_to_plot, width=1, color="darkblue"
                    )
                else:
                    g = sns.lineplot(
                        x=d["x"],
                        y=y_mean_and_std[:, 0],
                        ax=axs[0][axs_i],
                        linewidth=2,
                        color="darkblue",
                    )
                    g = axs[0][axs_i].fill_between(
                        x=d["x"],
                        y1=y_mean_and_std[:, 0] - y_mean_and_std[:, 1],
                        y2=y_mean_and_std[:, 0] + y_mean_and_std[:, 1],
                        color="aqua",
                    )
                    axs[1][axs_i].bar(
                        d["hist"][1][:-1], d["hist"][0], width=1, color="darkblue"
                    )
                axs[0][axs_i].axhline(y=0, color="grey", linestyle="--")
                axs[0][axs_i].set_title(
                    "{}:\n{:.2f}%".format(
                        self.bags[0]._split_long_titles(d["name"]), d["avg_effect"]
                    )
                )
                axs[0][axs_i].grid()
            axs_i += 1

        if show_n == 1:
            axs[1].get_xaxis().set_visible(False)
            axs[1].get_yaxis().set_visible(False)
        else:
            for i in range(show_n):
                axs[1][i].get_xaxis().set_visible(False)
                axs[1][i].get_yaxis().set_visible(False)
        plt.show()


if __name__ == "__main__":
    from sklearn.datasets import make_circles, make_regression

    """
    X_small, y_small = make_circles(n_samples=(250, 500), random_state=3, noise=0.04, factor=0.3)
    X_large, y_large = make_circles(n_samples=(250, 500), random_state=3, noise=0.04, factor=0.7)

    y_small[y_small == 1] = 0

    df = pd.DataFrame(np.vstack([X_small, X_large]), columns=['x1', 'x2'])
    df['label'] = np.hstack([y_small, y_large])
    df.label = 2 * df.label - 1

    sns.scatterplot(data=df, x='x1', y='x2', hue='label')
    df['x1'] = 1 * (df.x1 > 0)

    m = IGANN(n_estimators=100, n_hid=10, elm_alpha=5, boost_rate=1, sparse=0, verbose=2)
    start = time.time()

    inputs = df[['x1', 'x2']]
    targets = df.label

    m.fit(inputs, targets)
    end = time.time()
    print(end - start)

    m.plot_learning()
    m.plot_single(show_n=7)

    m.predict(inputs)

    ######
    """
    X, y = make_regression(1000, 4, n_informative=4, random_state=42)
    X = pd.DataFrame(X)
    X["cat_test"] = np.random.choice(
        ["A", "B", "C", "D"], X.shape[0], p=[0.2, 0.2, 0.1, 0.5]
    )
    X["cat_test_2"] = np.random.choice(
        ["E", "F", "G", "H"], X.shape[0], p=[0.2, 0.2, 0.1, 0.5]
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std
    start = time.time()
    m = IGANN_Bagged(
        task="regression", n_estimators=100, verbose=0, n_bags=5
    )  # , device='cuda'
    # m = IGANN(task='regression', n_estimators=100, verbose=0)
    m.fit(pd.DataFrame(X_train), y_train)
    end = time.time()
    print(end - start)
    m.plot_single(show_n=6, max_cat_plotted=4)

    #####
    """
    X, y = make_regression(10000, 2, n_informative=2)
    y = (y - y.mean()) / y.std()
    X = pd.DataFrame(X)
    X['categorical'] = np.random.choice(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], size=len(X))
    X['categorical2'] = np.random.choice(['q', 'b', 'c', 'd'], size=len(X), p=[0.1, 0.2, 0.5, 0.2])
    print(X.dtypes)
    m = IGANN(task='regression', n_estimators=100, sparse=0, verbose=2)
    m.fit(X, y)

    from Benchmark import FiveFoldBenchmark
    m = IGANN(n_estimators=0, n_hid=10, elm_alpha=5, boost_rate=1.0, sparse=0, verbose=2)
    #m = LogisticRegression()
    benchmark = FiveFoldBenchmark(model=m)
    folds_auroc = benchmark.run_model_on_dataset(dataset_id=1)
    print(np.mean(folds_auroc))
    """
