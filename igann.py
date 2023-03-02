import time
import torch
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import abess.linear

class torch_Ridge():
    def __init__(self, alpha, device):
        self.coef_ = None
        self.alpha = alpha
        self.device = device

    def fit(self, X, y):
        self.coef_ = torch.linalg.solve(X.T @ X + self.alpha * torch.eye(X.shape[1]).to(self.device), X.T @ y)

    def predict(self, X):
        return torch.dot(X.to(self.device), self.coef_)


class ELM_Regressor():
    '''
    This class represents one single hidden layer neural network for a regression task.
    Trainable parameters are only the parameters from the output layer. The parameters
    of the hidden layer are sampled from a normal distribution. This increases the training
    performance significantly as it reduces the training task to a regularized linear
    regression (Ridge Regression), see "Extreme Learning Machines" for more details.
    '''

    def __init__(self, n_input, n_hid, seed=0, scale=10, elm_alpha=0.0001, act='elu',
                 device='cpu'):
        '''
        Input parameters:
        - n_input: number of inputs/features (should be X.shape[1])
        - n_hid: number of hidden neurons for the base functions
        - seed: This number sets the seed for generating the random weights. It should
                be different for each regressor
        - scale: the scale which is used to initialize the weights in the hidden layer of the
                 model. These weights are not changed throughout the optimization.
        - elm_alpha: the regularization of the ridge regression.
        - act: the activation function in the model. can be 'elu', 'relu' or a torch activation function.
        - device: the device on which the regressor should train. can be 'cpu' or 'cuda'.
        '''
        super().__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)
        # The following are the random weights in the model which are not optimized.
        self.hidden_list = torch.normal(mean=torch.zeros(n_input, n_input * n_hid), std=scale).to(device)

        mask = torch.block_diag(*[torch.ones(n_hid)] * n_input).to(device)
        self.hidden_mat = self.hidden_list * mask

        self.output_model = None
        self.n_input = n_input
        self.n_hid = n_hid
        self.scale = scale
        self.elm_alpha = elm_alpha
        if act == 'elu':
            self.act = torch.nn.ELU()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        else:
            self.act = act
        self.device = device

    def get_hidden_values(self, X):
        '''
        This step computes the values in the hidden layer. For this, we iterate
        through the input features and multiply the feature values with the weights
        from hidden_list. After applying the activation function, we return the result
        in X_hid
        '''
        X_hid = X @ self.hidden_mat
        X_hid = self.act(X_hid)

        return X_hid

    def predict(self, X, hidden=False):
        '''
        This function makes a full prediction with the model for a given input X.
        '''
        if hidden:
            X_hid = X
        else:
            X_hid = self.get_hidden_values(X)

        # Now, we can use the values in the hidden layer to make the prediction with
        # our ridge regression
        out = X_hid @ self.output_model.coef_
        return out

    def predict_single(self, x, i):
        '''
        This function computes the partial output of one base function for one feature.
        Note, that the bias term is not used for this prediction.
        Input parameters:
        x: a vector representing the values which are used for feature i
        i: the index of the feature that should be used for the prediction
        '''

        # See self.predict for the description - it's almost equivalent.
        x_in = x.reshape(len(x), 1)
        x_in = x_in @ self.hidden_mat[i, i * self.n_hid:(i + 1) * self.n_hid].unsqueeze(0)
        x_in = self.act(x_in)
        out = x_in @ self.output_model.coef_[i * self.n_hid: (i + 1) * self.n_hid].unsqueeze(1)
        return out

    def fit(self, X, y, mult_coef):
        '''
        This function fits the ELM on the training data (X, y).
        '''
        X_hid = self.get_hidden_values(X)
        X_hid_mult = X_hid * mult_coef
        # Fit the ridge regression on the hidden values.
        m = torch_Ridge(alpha=self.elm_alpha, device=self.device)
        m.fit(X_hid_mult, y)
        self.output_model = m
        return X_hid


class IGANN:
    '''
    This class represents the IGANN model. It can be used like a
    sklearn model (i.e., it includes .fit, .predict, .predict_proba, ...).
    The model can be used for a regression task or a binary classification task.
    For binary classification, the labels must be set to -1 and 1 (Note that labels with
    0 and 1 are transformed automatically). The model first fits a linear model and then
    subsequently adds ELMs according to a boosting framework.
    '''

    def __init__(self, task='classification', n_hid=10, n_estimators=5000, boost_rate=0.1, init_reg=1,
                 elm_scale=1, elm_alpha=1, sparse=0, act='elu', early_stopping=50, device='cpu',
                 random_state=1, optimize_threshold=False, verbose=0):
        '''
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
        '''
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
        self.regressors = []
        self.boosting_rates = []
        self.train_scores = []
        self.val_scores = []
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.regressor_predictions = []
        self.boost_rate = boost_rate
        self.target_remapped_flag = False
        '''Is set to true during the fit method if the target (y) is remapped to -1 and 1 instead of 0 and 1.'''

        if task == 'classification':
            # todo: torch
            self.init_classifier = LogisticRegression(penalty='l1', solver='liblinear', C=1 / self.init_reg,
                                                      random_state=random_state)
            self.criterion = lambda prediction, target: torch.nn.BCEWithLogitsLoss()(prediction,
                                                                                     torch.nn.ReLU()(target))
        elif task == 'regression':
            # todo: torch
            self.init_classifier = Lasso(alpha=self.init_reg)
            self.criterion = torch.nn.MSELoss()
        else:
            print('Task not implemented. Can be classification or regression')

    def _clip_p(self, p):
        if torch.max(p) > 100 or torch.min(p) < -100:
            warnings.warn(
                'Cutting prediction to [-100, 100]. Did you forget to scale y? Consider higher regularization elm_alpha.')
            return torch.clip(p, -100, 100)
        else:
            return p

    def _clip_p_numpy(self, p):
        if np.max(p) > 100 or np.min(p) < -100:
            warnings.warn(
                'Cutting prediction to [-100, 100]. Did you forget to scale y? Consider higher regularization elm_alpha.')
            return np.clip(p, -100, 100)
        else:
            return p

    def _loss_sqrt_hessian(self, y, p):
        '''
        This function computes the square root of the hessians of the log loss or the mean squared error.
        '''
        if self.task == 'classification':
            return 0.5 / torch.cosh(0.5 * y * p)
        else:
            return torch.sqrt(torch.tensor([2.0]).to(self.device))

    def _get_y_tilde(self, y, p):
        if self.task == 'classification':
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

    def fit(self, X, y, val_set=None, eval=None, plot_fixed_features=None):
        '''
        This function fits the model on training data (X, y).
        Parameters:
        X: the feature matrix
        y: the targets
        val_set: can be tuple (X_val, y_val) for a defined validation set. If not set,
        it will be split from the training set randomly.
        eval: can be tuple (X_test, y_test) for additional evaluation during training
        plot_fixed_features: Per default the most important features are plotted for verbose=2.
        This can be changed here to keep track of the same feature throughout training.
        '''

        self._reset_state()

        if type(X) == pd.DataFrame:
            self.feature_names = X.columns
            X = X.values
        else:
            self.feature_names = np.arange(X.shape[1]).astype(str)

        if type(y) == pd.Series:
            y = y.values

        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y.squeeze()).float()

        if self.task == 'classification':
            # In the case of targets in {0,1}, transform them to {-1,1} for optimization purposes
            if torch.min(y) != -1:
                self.target_remapped_flag = True
                y = 2 * y - 1

        if self.sparse > 0:
            feature_indizes = self._select_features(X, y)
            self.feature_names = np.array([f for e, f in enumerate(self.feature_names) if e in feature_indizes])
            X = X[:, feature_indizes]
            self.feature_indizes = feature_indizes
        else:
            self.feature_indizes = np.arange(X.shape[1])

        # Fit the linear model on all data
        self.init_classifier.fit(X, y)

        # Split the data into train and validation data and compute the prediction of the
        # linear model. For regression this is straightforward, for classification, we
        # work with the logits and not the probabilities. That's why we multiply X with
        # the coefficients and don't use the predict_proba function.
        if self.task == 'classification':
            if val_set == None:
                if self.verbose >= 1:
                    print('Splitting data')
                X, X_val, y, y_val = train_test_split(X, y, test_size=0.15, stratify=y, random_state=self.random_state)
            else:
                X_val = val_set[0]
                y_val = val_set[1]

            y_hat = torch.squeeze(
                torch.from_numpy(self.init_classifier.coef_.astype(np.float32)) @ torch.transpose(X, 0, 1)) + float(
                self.init_classifier.intercept_)
            y_hat_val = torch.squeeze(
                torch.from_numpy(self.init_classifier.coef_.astype(np.float32)) @ torch.transpose(X_val, 0, 1)) + float(
                self.init_classifier.intercept_)

        else:
            if val_set == None:
                if self.verbose >= 1:
                    print('Splitting data')
                X, X_val, y, y_val = train_test_split(X, y, test_size=0.15, random_state=self.random_state)
            else:
                X_val = val_set[0]
                y_val = val_set[1].squeeze()

            y_hat = torch.from_numpy(self.init_classifier.predict(X).squeeze().astype(np.float32))
            y_hat_val = torch.from_numpy(self.init_classifier.predict(X_val).squeeze().astype(np.float32))

        # Store some information about the dataset which we later use for plotting.
        self.X_min = list(X.min(axis=0))
        self.X_max = list(X.max(axis=0))
        self.unique = [torch.unique(X[:, i]) for i in torch.arange(X.shape[1])]
        self.hist = [torch.histogram(X[:, i]) for i in range(X.shape[1])]

        if self.verbose >= 1:
            print('Training shape: {}'.format(X.shape))
            print('Validation shape: {}'.format(X_val.shape))
            print('Regularization: {}'.format(self.init_reg))

        train_loss_init = self.criterion(y_hat, y)
        val_loss_init = self.criterion(y_hat_val, y_val)

        if self.verbose >= 1:
            print('Train: {:.4f} Val: {:.4f} {}'.format(train_loss_init, val_loss_init, 'init'))

        X, y, y_hat, X_val, y_val, y_hat_val = X.to(self.device), y.to(self.device), y_hat.to(self.device), X_val.to(
            self.device), y_val.to(self.device), y_hat_val.to(self.device)

        self._run_optimization(X, y, y_hat, X_val, y_val, y_hat_val, eval,
                               val_loss_init, plot_fixed_features)

        if self.task == 'classification' and self.optimize_threshold:
            self.best_threshold = self._optimize_classification_threshold(X, y)

        return

    def _run_optimization(self, X, y, y_hat, X_val, y_val, y_hat_val, eval,
                          best_loss, plot_fixed_features):
        '''
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
        plot_fixed_features: Per default the most important features are plotted for verbose=2.
        This can be changed here to keep track of the same feature throughout training.
        '''

        counter_no_progress = 0
        best_iter = 0

        # Sequentially fit one ELM after the other. Max number is stored in self.n_estimators.
        for counter in range(self.n_estimators):
            hessian_train_sqrt = self._loss_sqrt_hessian(y, y_hat)
            y_tilde = torch.sqrt(torch.tensor(0.5).to(self.device)) * self._get_y_tilde(y, y_hat)

            # Init ELM
            regressor = ELM_Regressor(n_input=X.shape[1], n_hid=self.n_hid,
                                      seed=counter, scale=self.elm_scale,
                                      elm_alpha=self.elm_alpha,
                                      act=self.act, device=self.device)

            # Fit ELM regressor
            X_hid = regressor.fit(X, y_tilde,
                                  torch.sqrt(torch.tensor(0.5).to(self.device)) * self.boost_rate * hessian_train_sqrt[
                                                                                                    :, None])

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
                self._print_results(counter, counter_no_progress, eval, self.boost_rate,
                                    train_loss, val_loss)

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
                print(f'Cutting at {best_iter}')
            self.regressors = self.regressors[:best_iter]
            self.boosting_rates = self.boosting_rates[:best_iter]

        return best_loss

    def _select_features(self, X, y):
        regressor = ELM_Regressor(X.shape[1], self.n_hid, seed=0,
                                  scale=self.elm_scale, act=self.act, device='cpu')
        X_tilde = regressor.get_hidden_values(X)
        groups = np.array([np.ones(self.n_hid) * i + 1 for i in range(X.shape[1])]).flatten()

        if self.task == 'classification':
            m = abess.linear.LogisticRegression(path_type='gs', cv=3, s_max=self.sparse, thread=0)
            m.fit(X_tilde.numpy(), np.where(y.numpy() == -1, 0, 1), group=groups)
        else:
            m = abess.linear.LinearRegression(path_type='gs', cv=3, s_max=self.sparse, thread=0)
            m.fit(X_tilde.numpy(), y, group=groups)
        
        active_features = np.where(np.sum(m.coef_.reshape(-1, self.n_hid), axis=1) != 0)[0]

        if self.verbose > 0:
            print(f'Found features {active_features}')

        return active_features

    def _print_results(self, counter, counter_no_progress, eval, boost_rate, train_loss, val_loss):
        '''
        This function plots our results.
        '''
        if counter_no_progress == 0:
            new_best_symb = '*'
        else:
            new_best_symb = ''
        if eval:
            test_pred = self.predict_raw(eval[0])
            test_loss = self.criterion(test_pred, eval[1])
            self.test_losses.append(test_loss)
            print(
                '{}{}: BoostRate: {:.3f}, Train loss: {:.5f} Val loss: {:.5f} Test loss: {:.5f}'.format(
                    new_best_symb, counter, boost_rate, train_loss, val_loss, test_loss))
        else:
            print(
                '{}{}: BoostRate: {:.3f}, Train loss: {:.5f} Val loss: {:.5f}'.format(
                    new_best_symb, counter, boost_rate, train_loss, val_loss))

    def _optimize_classification_threshold(self, X_train, y_train):
        '''
        This function optimizes the classification threshold for the training set for later predictions.
        The use of the function is triggered by setting the parameter optimize_threshold to True.
        This is one method which does the job. However, we noticed that it is not always the best method and hence it
        defaults to no threshold optimization.
        '''

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
        '''
        Similarly to sklearn, this function returns a matrix of the same length as X and two columns.
        The first column denotes the probability of class -1, and the second column denotes the
        probability of class 1.
        '''
        if self.task == 'regression':
            warnings.warn('The call of predict_proba for a regression task was probably incorrect.')

        pred = self.predict_raw(X)
        pred = self._clip_p_numpy(pred)
        pred = 1 / (1 + np.exp(-pred))

        ret = np.zeros((len(X), 2), dtype=np.float32)
        ret[:, 1] = pred
        ret[:, 0] = 1 - pred

        return ret

    def predict(self, X):
        '''
        This function returns a prediction for a given feature matrix X.
        Note: for a classification task, it returns the binary target values in a 1-d np.array, it can hold -1 and 1.
        If optimize_threshold is True for a classification task, the threshold is optimized on the training data.
        '''
        if self.task == 'regression':
            return self.predict_raw(X)
        else:
            pred_raw = self.predict_raw(X)
            # detach and numpy pred_raw
            if self.optimize_threshold:
                threshold = self.best_threshold
            else:
                threshold = 0
            pred = np.where(pred_raw < threshold, np.ones_like(pred_raw) * -1, np.ones_like(pred_raw)).squeeze()

            if self.target_remapped_flag:
                pred = np.where(pred == -1, 0, 1)

            return pred

    def predict_raw(self, X):
        '''
        This function returns a prediction for a given feature matrix X.
        Note: for a classification task, it returns the raw logit values.
        '''
        if type(X) == pd.DataFrame:
            X = np.array(X)
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float().to(self.device)
        X = X[:, self.feature_indizes]

        pred_nn = torch.zeros(len(X), dtype=torch.float32).to(self.device)
        for boost_rate, regressor in zip(self.boosting_rates, self.regressors):
            pred_nn += boost_rate * regressor.predict(X).squeeze()
        pred_nn = pred_nn.detach().cpu().numpy()
        X = X.detach().cpu().numpy()
        pred = pred_nn + (self.init_classifier.coef_.astype(np.float32) @ X.transpose()).squeeze() + self.init_classifier.intercept_

        return pred

    def _split_long_titles(self, l):
        return '\n'.join(l[p:p + 22] for p in range(0, len(l), 22))

    def get_shape_functions_as_dict(self):
        feature_effects = []
        for i, feat_name in enumerate(self.feature_names):
            feat_values = self.unique[i]
            if self.task == 'classification':
                pred = self.init_classifier.coef_[0, i] * feat_values
            else:
                pred = self.init_classifier.coef_[i] * feat_values
            feat_values = feat_values.to(self.device)
            for regressor, boost_rate in zip(self.regressors, self.boosting_rates):
                pred += (boost_rate * regressor.predict_single(feat_values.reshape(-1, 1), i).squeeze()).cpu()
            feature_effects.append(
                {'name': feat_name, 'x': feat_values.cpu(),
                 'y': pred, 'avg_effect': torch.mean(torch.abs(pred)),
                 'hist': self.hist[i]})

        overall_effect = np.sum([d['avg_effect'] for d in feature_effects])
        for d in feature_effects:
            d['avg_effect'] = d['avg_effect'] / overall_effect * 100

        return feature_effects

    def plot_single(self, plot_by_list=None, show_n=5, scaler_dict=None):
        '''
        This function plots the most important shape functions.
        Parameters:
        show_n: the number of shape functions that should be plotted.
        scaler_dict: dictionary that maps every numerical feature to the respective (sklearn) scaler.
                     scaler_dict[num_feature_name].inverse_transform(...) is called if scaler_dict is not None
        '''
        feature_effects = self.get_shape_functions_as_dict()
        if plot_by_list is None:
            top_k = [d for d in sorted(feature_effects, reverse=True, key=lambda x: x['avg_effect'])][:show_n]
            show_n = min(show_n, len(top_k))
        else:
            top_k = [d for d in sorted(feature_effects, reverse=True, key=lambda x: x['avg_effect'])]
            show_n = len(plot_by_list)

        plt.close(fig="Shape functions")
        fig, axs = plt.subplots(2, show_n, figsize=(14, 4),
                                gridspec_kw={'height_ratios': [5, 1]},
                                num="Shape functions")
        plt.subplots_adjust(wspace=0.4)

        i = 0
        for d in top_k:
            if plot_by_list is not None and d['name'] not in plot_by_list:
                continue
            if scaler_dict:
                d['x'] = scaler_dict[d['name']].inverse_transform(d['x'].reshape(-1, 1)).squeeze()
            if len(d['x']) < 4:
                if show_n == 1:
                    sns.barplot(x=d['x'].numpy(), y=d['y'].numpy(), color="darkblue", ax=axs[0])
                    axs[1].bar(d['hist'][1][:-1], d['hist'][0], width=1, color='darkblue')
                    axs[0].set_title('{}:\n{:.2f}%'.format(self._split_long_titles(d['name']),
                                                           d['avg_effect']))
                    axs[0].grid()
                else:
                    sns.barplot(x=d['x'].numpy(), y=d['y'].numpy(), color="darkblue", ax=axs[0][i])
                    axs[1][i].bar(d['hist'][1][:-1], d['hist'][0], width=1, color='darkblue')
                    axs[0][i].set_title('{}:\n{:.2f}%'.format(self._split_long_titles(d['name']),
                                                              d['avg_effect']))
                    axs[0][i].grid()

            else:
                if show_n == 1:
                    g = sns.lineplot(x=d['x'].numpy(), y=d['y'].numpy(), ax=axs[0], linewidth=2, color="darkblue")
                    g.axhline(y=0, color="grey", linestyle="--")
                    axs[1].bar(d['hist'][1][:-1], d['hist'][0], width=1, color='darkblue')
                    axs[0].set_title('{}:\n{:.2f}%'.format(self._split_long_titles(d['name']),
                                                           d['avg_effect']))
                    axs[0].grid()
                else:
                    g = sns.lineplot(x=d['x'].numpy(), y=d['y'].numpy(), ax=axs[0][i], linewidth=2, color="darkblue")
                    g.axhline(y=0, color="grey", linestyle="--")
                    axs[1][i].bar(d['hist'][1][:-1], d['hist'][0], width=1, color='darkblue')
                    axs[0][i].set_title('{}:\n{:.2f}%'.format(self._split_long_titles(d['name']),
                                                              d['avg_effect']))
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
        '''
        Plot the training and the validation losses over time (i.e., for the sequence of learned
        ELMs)
        '''
        fig, axs = plt.subplots(1, 1, figsize=(16, 8))
        fig.axes[0].plot(np.arange(len(self.train_losses)), self.train_losses, label='Train')
        fig.axes[0].plot(np.arange(len(self.val_losses)), self.val_losses, label='Val')
        if len(self.test_losses) > 0:
            fig.axes[0].plot(np.arange(len(self.test_losses)), self.test_losses, label='Test')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    from sklearn.datasets import make_circles, make_regression
    
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
    
    X, y = make_regression(100000, 10, n_informative=3)
    y = (y - y.mean()) / y.std()
    m = IGANN(task='regression', sparse=5, verbose=2)
    m.fit(X, y)
    m.plot_learning()
    m.plot_single()
