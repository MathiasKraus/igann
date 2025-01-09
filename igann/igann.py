import time
import torch
import warnings
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

warnings.simplefilter("once", UserWarning)


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
        act="elu",
        early_stopping=50,
        device="cpu",
        random_state=1,
        verbose=0,
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
        act: the activation function in the ELM model. Can be 'elu', 'relu' or a torch activation function.
        early_stopping: we use early stopping which means that we don't continue training more ELM
        models, if there has been no improvements for 'early_stopping' number of iterations.
        device: the device on which the model is optimized. Can be 'cpu' or 'cuda'
        random_state: random seed.
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
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        self.boost_rate = boost_rate
        self.target_remapped_flag = False
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

    def _preprocess_feature_matrix(self, X, fit_transform=True):
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

        if len(self.categorical_cols) == 0:
            print("not cat columns detected")
            self.feature_names = self.numerical_cols
            self.n_numerical_cols = len(self.numerical_cols)
            self.n_categorical_cols = 0
            X_tensor = torch.tensor(X.values, dtype=torch.float32)

        else:
            # Define a ColumnTransformer
            if fit_transform:
                self.column_transformer = ColumnTransformer(
                    transformers=[
                        # (
                        #    "num",
                        #    #StandardScaler(), #todo: add this and then reverse in plotting
                        #    self.numerical_cols,
                        # ),
                        (
                            "cat",
                            OneHotEncoder(
                                drop="first",
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                            self.categorical_cols,
                        ),
                    ],
                    remainder="passthrough",  # Keep other columns, if any
                    verbose_feature_names_out=False,  # persevre column names after one hot encoding
                ).set_output(transform="pandas")
                # Fit and transform the data
                X_transformed = self.column_transformer.fit_transform(X)
            else:
                # Transform using the pre-fitted ColumnTransformer
                X_transformed = self.column_transformer.transform(X)

            # Record feature names for reference
            self.feature_names = self.numerical_cols + list(
                self.column_transformer.named_transformers_[
                    "cat"
                ].get_feature_names_out(self.categorical_cols)
            )

            # Identify dropped features from OneHotEncoder
            one_hot_encoder = self.column_transformer.named_transformers_["cat"]
            self.dropped_features = {
                feature: categories[0]
                for feature, categories in zip(
                    self.categorical_cols, one_hot_encoder.categories_
                )
            }

            # Set the number of categorical and numerical columns based on OneHotEncoder output
            self.n_numerical_cols = len(self.numerical_cols)
            self.n_categorical_cols = len(
                self.column_transformer.named_transformers_[
                    "cat"
                ].get_feature_names_out(self.categorical_cols)
            )

            # Reorder X to match feature_names list (we often need this for itterating through the features):
            # safty check if all columns are present
            missing = [
                col for col in self.feature_names if col not in X_transformed.columns
            ]
            if missing:
                raise ValueError(
                    f"DataFrame is missing columns needed for ordering: {missing}"
                )

            X_transformed = X_transformed[self.feature_names]

            # Convert to PyTorch tensor
            X_tensor = torch.tensor(X_transformed.to_numpy(), dtype=torch.float32)

        return X_tensor

    def fit(self, X, y, val_set=None):
        """
        This function fits the model on training data (X, y).
        Parameters:
        X: the feature matrix
        y: the targets
        val_set: can be tuple (X_val, y_val) for a defined validation set. If not set,
        it will be split from the training set randomly.
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
            "device": self.device,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "boost_rate": self.boost_rate,
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

    def _run_optimization(self, X, y, y_hat, X_val, y_val, y_hat_val, best_loss):
        """
        This function runs the optimization for ELMs with single features. This function should not be called from outside.
        Parameters:
        X: the training feature matrix
        y: the training targets
        y_hat: the current prediction for y
        X_val: the validation feature matrix
        y_val: the validation targets
        y_hat_val: the current prediction for y_val
        best_loss: best previous loss achieved. This is to keep track of the overall best sequence of ELMs.
        """

        counter_no_progress = 0
        best_iter = 0

        # Sequentially fit one ELM after the other. Max number is stored in self.n_estimators.
        for counter in range(self.n_estimators):
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
                    self.boost_rate,
                    train_loss,
                    val_loss,
                )

            # Stop training if the counter for early stopping is greater than the parameter we passed.
            if counter_no_progress > self.early_stopping and self.early_stopping > 0:
                break

            if self.verbose >= 2:
                if counter % 5 == 0:
                    self.plot_single()

        if self.early_stopping > 0:
            # We remove the ELMs that did not improve the performance. Most likely best_iter equals self.early_stopping.
            if self.verbose > 0:
                print(f"Cutting at {best_iter}")
            self.regressors = self.regressors[:best_iter]
            self.boosting_rates = self.boosting_rates[:best_iter]

        return best_loss

    def _print_results(
        self, counter, counter_no_progress, boost_rate, train_loss, val_loss
    ):
        """
        This function plots our results.
        """
        if counter_no_progress == 0:
            new_best_symb = "*"
        else:
            new_best_symb = ""
        print(
            "{}{}: BoostRate: {:.3f}, Train loss: {:.5f} Val loss: {:.5f}".format(
                new_best_symb, counter, boost_rate, train_loss, val_loss
            )
        )

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
        """
        if self.task == "regression":
            return self.predict_raw(X)
        else:
            pred_raw = self.predict_raw(X)
            # detach and numpy pred_raw
            pred = np.where(
                pred_raw < 0,
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
        X = self._preprocess_feature_matrix(X, fit_transform=False).to(self.device)

        pred_nn = torch.zeros(len(X), dtype=torch.float32).to(self.device)
        for boost_rate, regressor in zip(self.boosting_rates, self.regressors):
            pred_nn += boost_rate * regressor.predict(X).squeeze()
        pred_nn = pred_nn.detach().cpu().numpy()
        X = X.detach().cpu().numpy()
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
        for i, feat_name in enumerate(self.feature_names):
            # set datatype to numerical if i is smaller than the number of numerical columns
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
                        "hist": {
                            # make this list for eaysier handling and plotting
                            "counts": self.hist[i].hist.cpu().tolist(),
                            "edges": self.hist[i].bin_edges.cpu().tolist(),
                        },
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
                        "hist": {
                            "counts": [self.hist[i][0][-1].cpu().tolist()],
                            "classes": [class_name],
                        },
                    }
                )

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
                final_shape_functions[name]["hist"]["counts"].extend(
                    shape_function["hist"]["counts"]
                )
                final_shape_functions[name]["hist"]["classes"].extend(
                    shape_function["hist"]["classes"]
                )
            # if the feature is numerical or categorical and not in the dict yet we just add it to the final shape functions
            else:
                final_shape_functions[name] = shape_function

        # if we have dropped features we need to add them to the final shape functions
        num_rows = self.raw_X.shape[0]
        for name, function in final_shape_functions.items():
            if final_shape_functions[name]["datatype"] == "categorical":
                class_name = str(self.dropped_features[name])
                final_shape_functions[name]["x"].append(class_name)
                final_shape_functions[name]["y"].append(0)  # droped class effect is 0

                final_shape_functions[name]["hist"]["counts"].append(
                    num_rows - np.sum(final_shape_functions[name]["hist"]["counts"])
                )
                final_shape_functions[name]["hist"]["classes"].append(class_name)

        return final_shape_functions

    def plot_single(
        self,
        plot_by_list=None,
        show_n=5,
        scaler_dict=None,
        max_cat_plotted=4,
        max_plots_per_row=3,
    ):
        """ """
        # get shapefunctions
        shape_functions_raw = self.get_shape_functions_as_dict()

        # get names/keys to extract
        if plot_by_list is not None:
            show_n = len(plot_by_list)
            keys = plot_by_list
        else:
            keys = shape_functions_raw.keys()

        # convert to list
        shape_function_list = [shape_functions_raw[name] for name in keys]

        # sort shape functions by effect strength
        sorted_shape_functions = sorted(
            shape_function_list, reverse=True, key=lambda x: x["avg_effect"]
        )

        # redeuce list of shape function to required size
        top_k = sorted_shape_functions[:show_n]

        # set up a grid
        n_rows = int(np.ceil(len(top_k) / max_plots_per_row))
        n_cols = min(len(top_k), max_plots_per_row)

        # So the actual total rows = 2 * n_rows
        total_rows = 2 * n_rows
        total_cols = n_cols

        # create height ratios for the grid
        height_ratios = [4, 1] * n_rows  # shape is 4 histogram is 1

        # set up figure
        plt.close(fig="shape functions")
        fig, axs = plt.subplots(
            total_rows,
            total_cols,
            figsize=(12, 4 * n_rows),  # tune as you like
            gridspec_kw={
                "height_ratios": height_ratios,
                "hspace": 0.5,
                "wspace": 0.4,
            },
            # gridspec_kw={"height_ratios": [5, 1]},
            num="Shape functions",
        )

        # Force axs to be 2D if it is not already
        axs = axs.reshape(total_rows, total_cols)

        def _inverse_transform_x_if_needed(shape_func, scaler_dict):
            """
            Inversely transform the shape function's x and y values and histogram edges
            if:
            1) shape_func is numeric, AND
            2) shape_func['name'] is in scaler_dict
            """
            # If no scaler_dict is provided, just return as-is
            if scaler_dict is None:
                return shape_func

            # if y is in scaler_dict, inverse-transform
            if "y" in scaler_dict:
                scaler_func = scaler_dict["y"]
                shape_func["y"] = np.array(
                    scale_func(np.array(shape_func["y"]).reshape(-1, 1))
                )

            # Check if in scaler_dict
            if shape_func["name"] in scaler_dict:
                scaler_func = scaler_dict[shape_func["name"]]

                # Inverse-transform x-values
                x_arr = np.array(shape_func["x"]).reshape(-1, 1)
                x_inv = scaler_func(x_arr)
                shape_func["x"] = np.array(x_inv).ravel()

                # Inverse-transform histogram edges
                edges_arr = np.array(shape_func["hist"]["edges"]).reshape(-1, 1)
                edges_inv = scaler_func(edges_arr)
                shape_func["hist"]["edges"] = np.array(edges_inv).ravel()

            return shape_func

        # helper function to plot numerical shape
        def _plot_numeric(ax_top, ax_bottom, shape_function):
            shape_function = _inverse_transform_x_if_needed(shape_function, scaler_dict)
            # print(shape_function["x"])
            # print(shape_function["y"])
            # print(shape_function["hist"]["edges"])
            sns.lineplot(
                x=shape_function["x"],
                y=shape_function["y"],
                ax=ax_top,
                linewidth=3,
                color="darkblue",
            )
            ax_top.axhline(y=0, color="grey", linestyle="--")
            ax_bottom.bar(
                shape_function["hist"]["edges"][:-1],
                shape_function["hist"]["counts"],
                width=1,
                color="darkblue",
            )
            ax_bottom.get_xaxis().set_visible(False)

        # helper function to plot categorical shape
        def _plot_categorical(ax_top, ax_bottom, shape_function):
            shape_function = _inverse_transform_x_if_needed(shape_function, scaler_dict)
            ax_top.bar(
                x=shape_function["x"],
                height=shape_function["y"],
                width=1,
                color="darkblue",
            )
            ax_top.axhline(y=0, color="grey", linestyle="--")

            # set xticks and labels
            ax_top.set_xticks(np.arange(len(shape_function["x"])))
            ax_top.set_xticklabels(shape_function["x"], rotation=70)

            ax_bottom.bar(
                x=shape_function["hist"]["classes"],
                height=shape_function["hist"]["counts"],
                width=1,
                color="darkblue",
            )
            ax_bottom.get_xaxis().set_visible(False)

        # main loop
        # print(f"n_rows: {n_rows}, n_cols: {n_cols}")
        for i, shape_function in enumerate(top_k):
            # determine postion
            row = i // n_cols
            col = i % n_cols

            # print(f"i:{i}, row: {row}, col: {col}")

            # get axes
            ax_top = axs[2 * row, col]
            ax_bottom = axs[2 * row + 1, col]

            if shape_function["datatype"] == "numerical":
                _plot_numeric(ax_top, ax_bottom, shape_function)
                # Align x-axes for numeric only
                ax_bottom.set_xlim(ax_top.get_xlim())
            else:
                _plot_categorical(ax_top, ax_bottom, shape_function)

            # add a title
            ax_top.set_title(
                f"{shape_function['name']}: {shape_function['avg_effect']:.2f}"
            )

        # remove empty axes
        for i in range(show_n, n_cols * n_rows):
            row = i // n_cols
            col = i % n_cols
            axs[row * 2, col].axis("off")
            axs[row * 2 + 1, col].axis("off")

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

    m = IGANN(n_estimators=100, n_hid=10, elm_alpha=5, boost_rate=1, verbose=2)
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
    # m = IGANN_Bagged(
    #     task="regression", n_estimators=100, verbose=0, n_bags=5
    # )  # , device='cuda'
    m = IGANN(task="regression", n_estimators=100, verbose=0)
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
    m = IGANN(task='regression', n_estimators=100, verbose=2)
    m.fit(X, y)

    from Benchmark import FiveFoldBenchmark
    m = IGANN(n_estimators=0, n_hid=10, elm_alpha=5, boost_rate=1.0, verbose=2)
    #m = LogisticRegression()
    benchmark = FiveFoldBenchmark(model=m)
    folds_auroc = benchmark.run_model_on_dataset(dataset_id=1)
    print(np.mean(folds_auroc))
    """
