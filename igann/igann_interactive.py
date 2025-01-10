from igann import IGANN
from igann import ELM_Regressor

# not sure if we need everything here..... clean later!
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


class IGANN_interactive(IGANN):
    """
    Extends IGANN to use shape functions directly for predictions during both training
    and inference, enabling customizable and interpretable decision logic.

    Features:
    - **Shape Function Predictions:** Simplifies predictions by relying on shape functions
        instead of the ensemble, reducing complexity and accelerating inference.
    - **Interactive Customization:** Facilitates real-time adaptation of decision logic
        through modifiable shape functions.
    - **Training Flexibility:** Supports using shape functions during training, with a potenial benefit of decrasing
        memory usage for enhanced interpretability.
    - **IGANN Compatibility:** Closely mimics IGANN's behavior, allowing seamless integration.

    Benefits:
    - Faster predictions and reduced computational complexity.
    - Enhanced control and interpretability of decision logic.

    Note: This class is experimental and may not be well maintained.
    """

    def __init__(
        self, *args, GAMwrapper=True, GAM_detail=100, regressor_limit=100, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.GAMwrapper = GAMwrapper
        self.GAM = None
        self.GAM_detail = GAM_detail
        self.regressor_limit = regressor_limit

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
            #### Start - addtional code for IGANN_interactive #####
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
            #### Start - addtional code for IGANN_interactive #####
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
        # if we use the GAMwrapper, we compress the ELMs to a GAM model in the end of the optimization
        if self.GAMwrapper == True:
            self.compress_to_GAM()
        return best_loss

        return best_loss

    def predict_raw(self, X):
        """
        This function returns a prediction for a given feature matrix X.
        Note: for a classification task, it returns the raw logit values.
        """
        #### Start - addtional code for IGANN_interactive #####
        # if we have a GAM wrapper, we use the GAM model for prediction
        if self.GAMwrapper == True and self.GAM is not None:
            # Its not really correct to name this nn, but since pred_nn i further processed this makes sensense
            pred_nn = self.GAM.predict_raw(X)
            pred = pred_nn + (self.linear_model.intercept_)
        #### End - addtional code for IGANN_interactive #####
        else:
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

    def _get_pred_of_i(self, i, x_values=None):
        if x_values == None:
            feat_values = self.unique[i]
        else:
            feat_values = x_values[i]

        #### Start - addtional code for IGANN_interactive #####
        # if there is a GAMwarapper and its feature dict is set up we use this for a prediction
        if self.GAMwrapper and self.GAM and self.GAM.feature_dict:
            pred = self.GAM.predict_single(i, feat_values)
            pred = torch.from_numpy(np.array(pred))
        #### End - addtional code for IGANN_interactive #####

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

    #### Start - addtional code for IGANN_interactive #####
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

    #### End - addtional code for IGANN_interactive #####


class GAMmodel:
    """
    This is a wrapper class for the GAM model it handels the alternative functions that are based on the shapefunctions.
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
        return

    def update_feature_dict(self, feat_dict):
        self.feature_dict.update(feat_dict)
        return

    def set_shape_functions(self):
        """
        This function creates the shape functions for the GAM model.
        it simply call the IGANN function get_shape_functions_as_dict and then creates the shape functions for the GAM model.
        This might looks redundant but could be helpful if we want to use a different model for the shape functions.
        """
        # TODO: Check if we can use the Base IGANN Shapefunction without manipulating it.
        shape_data = self.base_model.get_shape_functions_as_dict()
        for feature, feature_dict in shape_data.items():

            feature_name = feature_dict["name"]
            feature_type = feature_dict["datatype"]
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
        """
        this function creates the points for the shape functions that are saved for numeric features.
        """
        min_x, max_x = min(X), max(X)
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
        # Some times we use a interger to get the feature name (not sure why)
        if type(feature_name) == int:
            feature_name = self.base_model.feature_names[feature_name]

        # If the feature is not in the feature dict try to handle it like a one-hot encoded one.
        if feature_name not in self.feature_dict.keys():
            # reconstuct original feature name
            new_feature_name = feature_name.rsplit("_", 1)[0]
            # extract new class name
            class_name = feature_name.rsplit("_", 1)[-1]
            # create the new feature
            x = [
                (
                    class_name
                    if x == 1
                    # we fill in the class name of the droped feature which results in y = 0
                    else self.base_model.dropped_features[new_feature_name]
                )
                for x in x
            ]
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
        y = {}
        for col in X.columns:
            y[col] = self.predict_single(col, X[col])

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
