import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge, LogisticRegression, Lasso
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error
from sklearn.model_selection import train_test_split
import scipy.optimize as optim
import matplotlib.pyplot as plt

class ELM_Regressor():
    '''
    This class represents one single hidden neural network for a regression task.
    Trainable parameters are only the parameters from the last layer, the other 
    parameters stay random. This is mostly to speed up training and to partially
    prevent overfitting. The name of this model is called 'extreme learning
    machine'. 
    For fitting, the model then only has to optimize a linear regression (Ridge in
    our case).
    See wikipedia and google for more details .
    '''
    def __init__(self, n_input, n_hid, seed=0, scale=10, elm_alpha=0.0001, act='elu'):
        '''
        Input parameters:
        - n_input: number of inputs/features for your task (shape[1] of your X matrix)
        - n_hid: number of hidden neurons for each feature. Multiply this number with n_input
        to get the number of parameters this model has (-1 because of the bias term)
        - seed: This number sets the seed for generating the random weights. It should
        be different for each regressor
        - scale: the scale which is used to initialize the weights in the first layer of the 
        model. These weights are not changed throughout the optimization. This parameter
        has huge impact on the model. A larger scale makes the shape functions sharper, a lower
        scale makes them smoother. Definitely play around with this one.
        - elm_alpha: the regularization of the ridge regression. I didn't find this to have
        a huge impact, but let's try it out...
        - act: the activation function in the model. I mainly used 'elu' which works well,
        'relu' is kind of a mess, 'gelu' also provides nice results.
        '''
        super().__init__()
        np.random.seed(seed)
        # The following are the random weights in the model which are not optimized. 
        self.hidden_list = [np.random.normal(scale=scale, size=(1, n_hid)) for _ in range(n_input)]
        self.output_model = None
        self.n_hid = n_hid
        self.scale = scale
        self.elm_alpha = elm_alpha
        self.rand_act = act

    def act(self, x):
        if self.rand_act == 'elu':
            x[x <= 0] = np.exp(x[x <= 0]) - 1
        elif self.rand_act == 'relu':
            x[x <= 0] = 0
        elif self.rand_act == 'gelu':
            x = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
        return x

    def get_hidden_values(self, X):
        '''
        This step computes the values in the hidden layer. For this, we iterate
        through the input features and multiple the feature values with the weights 
        from hidden_list. After applying the activation function, we return the result
        in X_hid
        '''
        X_hid = np.zeros((X.shape[0], X.shape[1] * self.n_hid))
        for i in range(X.shape[1]):
            x_in = np.expand_dims(X[:, i], 1)
            x_in = np.dot(x_in, self.hidden_list[i])
            x_in = self.act(x_in)
            X_hid[:, i * self.n_hid:(i + 1) * self.n_hid] = x_in
        return X_hid

    def predict(self, X):
        '''
        This function makes a full prediction with the model for a given input X.
        '''

        X_hid = self.get_hidden_values(X)

        # Now, we can use the values in the hidden layer to make the prediction with
        # our ridge regression
        out = self.output_model.predict(X_hid)
        return out

    def predict_single(self, x, i):
        '''
        This function computes the output of one shape function. Note, that the 
        bias term is not used for this prediction.
        Input parameters:
        x: a vector representing the values which are used for feature i
        i: the index of the feature that should be used for the prediction
        '''

        # See self.predict for the description - it's almost equivalent.
        x_in = x.reshape(len(x), 1)
        x_in = np.dot(x_in, self.hidden_list[i])
        x_in = self.act(x_in)
        out = np.dot(x_in, np.expand_dims(self.output_model.coef_[i * self.n_hid: (i + 1) * self.n_hid], 1))
        return out

    def fit(self, X, y):
        '''
        This function fits the ELM on the training data (X, y). 
        '''
        X_hid = self.get_hidden_values(X)

        # Fit the ridge regression on the hidden values.
        m = Ridge(alpha=self.elm_alpha)
        with sklearn.config_context(assume_finite=True):
            m.fit(X_hid, y)
        self.output_model = m


class ELM_Regressor_Feat_Pair():
    def __init__(self, input_pairs, n_hid, seed=0, scale=10, elm_alpha=0.0001, act='elu'):
        super().__init__()
        np.random.seed(seed)
        self.hidden_list = [(np.random.normal(scale=scale, size=(1, n_hid)),
                              np.random.normal(scale=scale, size=(1, n_hid))) for _ in range(len(input_pairs))] 

        self.input_pairs = input_pairs
        self.output_model = None
        self.n_hid = n_hid
        self.scale = scale
        self.elm_alpha = elm_alpha
        self.rand_act = act

    def act(self, x):
        if self.rand_act == 'elu':
            x[x <= 0] = np.exp(x[x <= 0]) - 1
        elif self.rand_act == 'relu':
            x[x <= 0] = 0
        elif self.rand_act == 'gelu':
            x = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
        return x

    def get_hidden_values(self, X):
        X_hid = np.zeros(shape=(X.shape[0], len(self.input_pairs) * self.n_hid))
        
        for c, (i, j) in enumerate(self.input_pairs):
            x_in = np.expand_dims(X[:, i], 1)
            x_in = np.dot(x_in, self.hidden_list[c][0])
            x_jn = np.expand_dims(X[:, j], 1)
            x_jn = np.dot(x_jn, self.hidden_list[c][1])
            
            X_hid[:, c * self.n_hid : (c + 1) * self.n_hid] = self.act(x_in + x_jn)

        return X_hid

    def predict(self, X):
        X_hid = self.get_hidden_values(X)

        out = self.output_model.predict(X_hid)
        return out

    def predict_single(self, x1, x2, i):
        x1 = x1.reshape(len(x1), 1)
        x2 = x2.reshape(len(x2), 1)
        
        x1 = np.dot(x1, self.hidden_list[i][0])
        x2 = np.dot(x2, self.hidden_list[i][1])
        
        x = self.act(x1 + x2)
        
        out = np.dot(x, np.expand_dims(self.output_model.coef_[i * self.n_hid: (i + 1) * self.n_hid], 1))
        return out

    def fit(self, X, y):
        X_hid = self.get_hidden_values(X)

        # Fit the ridge regression on the hidden values.
        m = Ridge(alpha=self.elm_alpha)
        with sklearn.config_context(assume_finite=True):
            m.fit(X_hid, y)
        self.output_model = m


class IGANN:
    '''
    This class represents our model igann. We can use it mostly like
    a normal sklearn model (using .fit, .predict, .predict_proba, ...). The model can be used for two 
    tasks --- regression and binary classification. For binary classification, the labels must be -1 
    and 1. 
    The model first fits a linear model and then subsequently, train ELMs on the gradients of the
    previous prediction (the boosting idea).
    '''
    def __init__(self, task='classification', n_hid=10, n_estimators=500, boost_rate='auto', init_reg=0.5, 
    		     elm_scale=10, elm_alpha=0.0001, interactions=0, n_hid_interactions=10, only_interactions=False,
                 act='elu', early_stopping=50, random_state=1,
                 verbose=1):
        '''
        Initialize the model. Input parameters:
        task: defines the task, can be 'regression' or 'classification'
        n_hid: the number of hidden neurons for one feature
        n_estimators: the maximum number of estimators (ELMs) to be fitted.
        boost_rate: the boosting rate with which the predictions are updated with a new ELM model.
        It can be a real number 0.5/1.0,... or 'auto'. With 'auto' the boosting_rate is optimized.
        init_reg: the initial regularization strength for the linear model.
        elm_scale: the scale of the random weights in the elm model.
        elm_alpha: the regularization strength for the ridge regression in the ELM model.
        interactions: the number of interactions that should be fit.
        n_hid_interactions: the number of hidden neurons for one pair of features (one interaction)
        only_interactions: optimize only interactions and no single-feature ELMs (default is False)
        act: the activation function in the ELM model.
        early_stopping: we use early stopping which means that we don't continue training more ELM 
        models, if there as been no improvements for 'early_stopping' number of iterations.
        random_state: should take the randomness away. Probably needs to be fixed - I am not good
        at this.
        verbose: tells how many information to be printed when fitting. Can be 0 for (almost) no 
        information, 1 for printed losses, and 2 for plotted shape functions in each iteration.
        '''
        self.task = task
        self.n_hid = n_hid
        self.boost_rate = boost_rate
        self.elm_scale = elm_scale
        self.elm_alpha = elm_alpha
        self.init_reg = init_reg
        self.act = act
        self.n_estimators = n_estimators
        self.early_stopping = early_stopping
        self.interactions = interactions
        self.only_interactions = only_interactions
        self.random_state = random_state
        self.verbose = verbose
        self.n_hid_interactions = n_hid_interactions
        self.regressors = []
        self.boosting_rates = []
        self.train_scores = []
        self.val_scores = []
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []

        if task == 'classification':
            self.init_classifier = LogisticRegression(penalty='l1', solver='liblinear', C=self.init_reg,
                                                      random_state=random_state)
        elif task == 'regression':
            self.init_classifier = Lasso(alpha=1/self.init_reg)
        else:
            print('Task not implemented. Can be classification or regression')

    def _loss_gradient(self, y, p):
        '''
        This function computes the gradients of the log loss or the mean squared error.
        '''
        if self.task == 'classification':
            return - y / (1 + np.exp(y * p))
        else:
            return p - y
        
    def _act(self, x):
        x[x <= 0] = np.exp(x[x <= 0]) - 1
        return x

    def fit(self, X, y, val_set=None, eval=None, plot_fixed_features=None):
        '''
        This function fits the model on training data (X, y). 
        Parameters:
        X: the feature matrix
        y: the targets
        val_set: can be tuple (X_val, y_val) for a defined validation set. If not set,
        it will be split from the training set randomly. (This can be helpful for tasks like 
        time series analysis)
        eval: can be tuple (X_test, y_test) to have a peek on the test performance
        plot_fixed_features: Usually, the most important features are plotted for verbose=2.
        This can be changed here to keep track of the same feature throughout training.
        '''
        if type(X) == pd.DataFrame:
            self.feature_names = X.columns
            X = X.values
        else:
            self.feature_names = np.arange(X.shape[1]).astype(str)

        if type(y) == pd.Series:
            y = y.values

        y = y.squeeze()

        if self.task == 'classification':
            if np.min(y) == 0:
                print('Labels must be -1 and 1')
                return

        # Fit the linear model on all data
        self.init_classifier.fit(X, y)

        print(self.init_classifier.coef_)

        # Split the data into train and validation data and compute the prediction of the
        # linear model. For regression this is straightforward, for classification, we 
        # want to the logits and not the probabilities. That's why we multiply X with 
        # the coefficients and don't use the predict_proba function.
        if self.task == 'classification':
            if val_set == None:
                X, X_val, y, y_val = train_test_split(X, y, test_size=0.15, stratify=y, random_state=self.random_state)
            else:
                X_val = val_set[0]
                y_val = val_set[1]
            y_hat = np.matmul(self.init_classifier.coef_, X.transpose()).squeeze() + float(
                self.init_classifier.intercept_)
            y_hat_val = np.matmul(self.init_classifier.coef_, X_val.transpose()).squeeze() + float(
                self.init_classifier.intercept_)
        else:
            if val_set == None:
                X, X_val, y, y_val = train_test_split(X, y, test_size=0.15, random_state=self.random_state)
            else:
                X_val = val_set[0]
                y_val = val_set[1]
            y_hat = self.init_classifier.predict(X).squeeze()
            y_hat_val = self.init_classifier.predict(X_val).squeeze()

        # Store some information about the dataset which we later use for plotting.
        self.X_min = list(X.min(axis=0))
        self.X_max = list(X.max(axis=0))
        self.unique = [np.unique(X[:, i]) for i in np.arange(X.shape[1])]
        self.hist = [np.histogram(X[:, i]) for i in range(X.shape[1])]

        if self.verbose >= 1:
            print('Training shape: {}'.format(X.shape))
            print('Validation shape: {}'.format(X_val.shape))
            print('Regularizatoin: {}'.format(self.init_reg))

        if self.task == 'classification':
            # The next line translates the raw logits to a probability. This will come over 
            # and over again throughout the script. We need this to call the log_loss
            # function.
            y_pred = 1 / (1 + np.exp(-y_hat))
            train_loss_init = log_loss(y, y_pred)
            y_val_pred = 1 / (1 + np.exp(-y_hat_val))
            val_loss_init = log_loss(y_val, y_val_pred)
        else:
            train_loss_init = mean_squared_error(y, y_hat)
            val_loss_init = mean_squared_error(y_val, y_hat_val)
        
        if self.verbose >= 1:
            print('Train: {:.4f} Val: {:.4f} {}'.format(train_loss_init, val_loss_init, 'init'))

        if self.only_interactions:
            best_val_loss = np.inf
        else:
            best_val_loss = self._run_optimization(X, y, y_hat, X_val, y_val, y_hat_val, eval,
                                                  val_loss_init, plot_fixed_features,
                                                  interactions=False, feat_pairs=None, n_prev_regressors=0)
            
        self.single_regressors = len(self.regressors)
        if self.interactions == 0:
            return 
        
        y_hat = self.predict(X)
        y_hat_val = self.predict(X_val)
        
        gradients = -self._loss_gradient(y, y_hat)
        self.feat_pairs = self._find_interactions(X, gradients)
        
        self._run_optimization(X, y, y_hat, X_val, y_val, y_hat_val, eval,
                              best_val_loss, plot_fixed_features, 
                              interactions=True, feat_pairs=self.feat_pairs, 
                              n_prev_regressors=self.single_regressors)

        return 

    def _run_optimization(self, X, y, y_hat, X_val, y_val, y_hat_val, eval, 
                         best_loss, plot_fixed_features, 
                         interactions=False, feat_pairs=None, n_prev_regressors=0):
        '''
        This function runs the optimization for ELMs with single features or ELMs with 
        pairs of features (interactions). This function should not be called from outside.
        Parameters:
        X: the training feature matrix
        y: the training targets
        y_hat: the current prediction for y
        X_val: the valudation feature matrix
        y_val: the validation targets
        y_hat_val: the current prediction for y_val
        eval: can be tuple (X_test, y_test) to have a peek on the test performance
        best_loss: best previous loss achieved. This is to keep track of the overall best sequence of ELMs.
        plot_fixed_features: Usually, the most important features are plotted for verbose=2.
        interactions: True if the ELMs should fit interactions, else False
        feat_pairs: list of feature pairs when fitting interactions
        n_prev_regressors: number of previous regressors before starting the optimization
        '''

        counter_no_progress = 0
        best_iter = 0
        # Sequentially fit one ELM after the other. Max number is stored in self.n_estimators.
        for counter in range(self.n_estimators):
            
            # Compute the gradients of the loss
            gradients = -self._loss_gradient(y, y_hat)

            # Fit an ELM on the gradients
            if interactions:
                regressor = ELM_Regressor_Feat_Pair(feat_pairs, n_hid=self.n_hid, seed=counter, scale=self.elm_scale, elm_alpha=self.elm_alpha,
                                           act=self.act)
            else:
                regressor = ELM_Regressor(n_input=X.shape[1], n_hid=self.n_hid, seed=counter, scale=self.elm_scale, elm_alpha=self.elm_alpha,
                                           act=self.act)
            
            regressor.fit(X, gradients)

            # Make a prediction of the ELM for the gradients of train and val
            train_gradients_pred = regressor.predict(X).squeeze()
            val_gradients_pred = regressor.predict(X_val).squeeze()

            # Optimize the boosting rate or take the passed boosting rate
            if self.task == 'classification':
                fun = lambda boost_rate: log_loss(y_val, 1 / (1 + np.exp(-(y_hat_val + boost_rate * val_gradients_pred))))
            else:
                fun = lambda boost_rate: mean_squared_error(y_val, y_hat_val + boost_rate * val_gradients_pred)

            if self.boost_rate == 'auto':
                opt_boost_rate = optim.minimize_scalar(fun, bounds=(0.05, 1.0), method='bounded')
                boost_rate = opt_boost_rate.x
            else:
                boost_rate = self.boost_rate

            # Update the prediction for training and validation data
            y_hat += boost_rate * train_gradients_pred
            y_hat_val += boost_rate * val_gradients_pred

            # Make predictions are computing performance metrics 
            if counter == 0:
                val_pred = self.predict(X_val)
                train_pred = self.predict(X)
            else:
                val_pred = self._update_predict(X_val, val_pred, boost_rate)
                train_pred = self._update_predict(X, train_pred, boost_rate)
            
            if self.task == 'classification':
                val_pred_prob = 1 / (1 + np.exp(-val_pred))
                train_pred_prob = 1 / (1 + np.exp(-train_pred))

                val_auc = roc_auc_score(y_val, val_pred_prob)
                val_loss = log_loss(y_val, val_pred_prob)

                train_auc = roc_auc_score(y, train_pred_prob)
                train_loss = log_loss(y, train_pred_prob)
            else:
                val_mse = mean_squared_error(y_val, val_pred)
                val_loss = val_mse
                train_mse = mean_squared_error(y, train_pred)
                train_loss = train_mse

            # Keep the ELM, the boosting rate and losses in lists, so 
            # we can later use them again.
            self.regressors.append(regressor)
            self.boosting_rates.append(boost_rate)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # This is the early stopping mechanism. If there was no improvement on the 
            # validation set, we increase a counter by 1. If there was an improvement,
            # we set it back to 0.
            counter_no_progress += 1
            if val_loss < best_loss:
                best_iter = counter + 1
                best_loss = val_loss
                counter_no_progress = 0

            if self.verbose >= 1:
                if self.task == 'classification':
                    self._print_results(counter, counter_no_progress, eval, boost_rate, train_auc,
                                       train_loss, val_auc, val_loss)
                else:
                    self._print_results(counter, counter_no_progress, eval, boost_rate, train_mse,
                                       train_loss, val_mse, val_loss)
                
            # Stop training if the counter for early stopping is greater than the parameter we passed.
            if counter_no_progress > self.early_stopping and self.early_stopping > 0:
                break

            if self.verbose >= 2:
                if interactions:
                    self.plot_interactions()
                else:
                    if plot_fixed_features != None:
                        self.plot_by_list(plot_fixed_features)
                    else:
                        self.plot_single()
            
        if self.early_stopping > 0:
            # We remove the ELMs that did not improve the performance. Most likely best_iter equals self.early_stopping.
            print(f'Cutting at {best_iter}') 
            self.regressors = self.regressors[:best_iter + n_prev_regressors]
            self.boosting_rates = self.boosting_rates[:best_iter + n_prev_regressors]
            
        return best_loss

    def _find_interactions(self, X, y):
        '''
        This function finds the most promising pair of features for predicting y. It does so
        by generating a large hidden layer consisting of hidden activations of ELMs, where each ELM
        is getting two inputs. Then, we train Lasso models with varying regularization strength,
        until all coefficients are zero except for the coefficients within one/two/three 'ELM block'. 
        Thereby the number of ELM blocks must be equal to the input parameter self.interactions.
        This function should not be called from outside.
        '''
        if len(X) > 10000:
            sample = np.random.choice(np.arange(len(X)), size=10000, replace=False)
            X = X[sample]
            y = y[sample]
        X_hid = np.zeros(shape=(X.shape[0], X.shape[1] * X.shape[1] * self.n_hid_interactions))
        hidden_list = [[(np.random.normal(scale=1, size=(1, self.n_hid_interactions)),
                 np.random.normal(scale=1, size=(1, self.n_hid_interactions))) for _ in range(X.shape[1])] for _ in range(X.shape[1])]

        for i in range(X.shape[1]):
            for j in range(X.shape[1]):
                if i == j:
                    continue
                # This is super stupid
                x_in = np.expand_dims(X[:, i], 1)
                x_in = np.dot(x_in, hidden_list[i][j][0])
                x_jn = np.expand_dims(X[:, j], 1)
                x_jn = np.dot(x_jn, hidden_list[i][j][1])
        
                X_hid[:, (i * X.shape[1] + j) * self.n_hid_interactions:(i * X.shape[1] + j + 1) * self.n_hid_interactions] = self._act(x_in + x_jn)

        lower_list = [1e-10]
        upper_list = [1000]
        alpha = 1
        found = False
        for _ in range(1000):
            lasso = Lasso(alpha=alpha)
            lasso.fit(X_hid, y)
            feat_pairs = self._get_feat_pairs(lasso.coef_, X)
            if len(feat_pairs) == self.interactions:
                found = True
                break
            elif len(feat_pairs) < self.interactions:
                upper_list.append(alpha)
                alpha = np.random.uniform(low=np.max(lower_list), high=np.min(upper_list))
            else:
                lower_list.append(alpha)
                alpha = np.random.uniform(low=np.max(lower_list), high=np.min(upper_list))
      
        if not found:
            print('Did not find num of interactions wanted!!!! EXIT')
            return
        
        else:
            if self.verbose > 0:
                print(f'Found interactions {feat_pairs}')
            
        return feat_pairs

    def _get_feat_pairs(self, coef, X):
        '''
        This function computes how many ELM blocks are corresponding to coefficients != 0.
        It then returns all feature pairs that correspond to these ELM blocks.
        '''
        feat_pairs = []
        for pos in np.where(coef != 0)[0]:
            block = int((pos - (pos % self.n_hid_interactions)) / self.n_hid_interactions)
            corr_i_feat = int(block / X.shape[1])
            corr_j_feat = block % X.shape[1]
            if corr_i_feat < corr_j_feat:
                if (corr_i_feat, corr_j_feat) not in feat_pairs:
                    feat_pairs.append((corr_i_feat, corr_j_feat))
            else:
                if (corr_j_feat, corr_i_feat) not in feat_pairs:
                    feat_pairs.append((corr_j_feat, corr_i_feat))
        
        return feat_pairs

    def _print_results(self, counter, counter_no_progress, eval, boost_rate, train_perf, train_loss,
                      val_perf, val_loss):
        '''
        This function simply plots our results.
        '''
        if counter_no_progress == 0:
            new_best_symb = '*'
        else:
            new_best_symb = ''
        if self.task == 'classification':
            if eval:
                test_pred = self.predict(eval[0])
                test_pred_prob = 1 / (1 + np.exp(-test_pred))

                test_loss = log_loss(eval[1], test_pred_prob)
                print(
                    '{}{}: BoostRate: {:.3f}, Train AUROC {:.4f}, loss: {:.5f} Val AUROC {:.4f}, loss: {:.5f} Test loss: {:.5f}'.format(
                        new_best_symb, counter, boost_rate, train_perf, train_loss, val_perf, val_loss, test_loss))
                self.test_losses.append(test_loss)
            else:
                print(
                    '{}{}: BoostRate: {:.3f}, Train AUROC {:.4f}, loss: {:.5f} Val AUROC {:.4f}, loss: {:.5f}'.format(
                        new_best_symb, counter, boost_rate, train_perf, train_loss, val_perf, val_loss))
        else:
            if eval:
                test_pred = self.predict(eval[0])
                test_mse = mean_squared_error(eval[1], test_pred)
                print(
                    '{}{}: BoostRate: {:.3f}, Train MSE {:.4f}, loss: {:.4f} Val MSE {:.4f}, loss: {:.4f} Test MSE: {:.4f}'.format(
                        new_best_symb, counter, boost_rate, train_perf, train_loss, val_perf, val_loss, test_mse))
                self.test_losses.append(test_mse)
            else:
                print('{}{}: BoostRate: {:.3f}, Train MSE {:.4f}, loss: {:.4f} Val MSE {:.4f}, loss: {:.4f}'.format(
                    new_best_symb, counter, boost_rate, train_perf, train_loss, val_perf, val_loss))


    def predict_proba(self, X):
        '''
        Similarly to sklearn, this function returns a matrix of the same length as X and two columns.
        The first column denotes the probability of class -1, and the second column denotes the
        probability of class 1.
        '''
        pred = self.predict(X)
        pred = 1 / (1 + np.exp(-pred))

        ret = np.zeros((len(X), 2), dtype=float)
        ret[:, 1] = pred
        ret[:, 0] = 1 - pred

        return ret

    def _update_predict(self, X, pred, boost_rate):
        '''
        This is a helper function that speeds up training when we pass an evaluation set.
        '''
        if type(X) == pd.DataFrame:
            X = np.array(X)
        pred_nn = np.zeros(len(X), dtype=np.float32)
        regressor = self.regressors[-1]
        pred_nn += boost_rate * regressor.predict(X).squeeze()
        pred += pred_nn

        return pred

    def predict(self, X):
        '''
        This function returns a prediction for a given feature matrix X.
        Note: for a classification task, this simply returns the raw logit values.
        '''
        if type(X) == pd.DataFrame:
            X = np.array(X)
        pred_nn = np.zeros(len(X), dtype=np.float32)
        for boost_rate, regressor in zip(self.boosting_rates, self.regressors):
            pred_nn += boost_rate * regressor.predict(X).squeeze()
        pred = pred_nn + np.matmul(self.init_classifier.coef_,
                                   X.transpose()).squeeze() + self.init_classifier.intercept_

        return pred

    def _split_long_titles(self, l):
        return '\n'.join(l[p:p + 22] for p in range(0, len(l), 22))

    def get_shape_functions_as_dict(self):
        feature_effects = []
        for i, feat_name in enumerate(self.feature_names):
            if len(self.unique[i]) < 4:
                linspac = self.unique[i]
                if self.task == 'classification':
                    pred = self.init_classifier.coef_[0, i] * linspac
                else:
                    pred = self.init_classifier.coef_[i] * linspac
                # print(pred)
                for regressor, boost_rate in zip(self.regressors, self.boosting_rates):
                    pred += (boost_rate * regressor.predict_single(linspac.reshape(-1, 1), i).squeeze())
                    # print(pred)
                feature_effects.append(
                    {'name': feat_name, 'x': linspac, 
                     'y': pred, 'avg_effect': np.mean(np.abs(pred)),
                     'hist': self.hist[i]})
            else:
                linspac = self.unique[i] #np.linspace(self.X_min[i], self.X_max[i], 10000)
                if self.task == 'classification':
                    pred = self.init_classifier.coef_[0, i] * linspac
                else:
                    pred = self.init_classifier.coef_[i] * linspac
                for regressor, boost_rate in zip(self.regressors, self.boosting_rates):
                    pred += boost_rate * regressor.predict_single(linspac.reshape(-1, 1), i).squeeze()
                feature_effects.append(
                    {'name': feat_name, 'x': linspac, 
                     'y': pred, 'avg_effect': np.mean(np.abs(pred)),
                     'hist': self.hist[i]})

        overall_effect = np.sum([d['avg_effect'] for d in feature_effects])
        for d in feature_effects:
            d['avg_effect'] = d['avg_effect'] / overall_effect * 100
            
        return feature_effects

    def plot_single(self, show_n=5):
        '''
        This function plots the most important shape functions. 
        Parameters:
        show_n: the number of shape functions that should be plotted (don't know if this works).
        '''
        feature_effects = self.get_shape_functions_as_dict()
        top_k = [d for d in sorted(feature_effects, reverse=True, key=lambda x: x['avg_effect'])][:show_n]

        y_min = np.inf
        y_max = -np.inf
        fig, axs = plt.subplots(2, show_n, figsize=(14, 4),
                                gridspec_kw={'height_ratios': [5, 1]})
        for i, d in enumerate(top_k):
            if len(d['x']) < 4:
                axs[0][i].scatter(d['x'], d['y'], c='gray')
                axs[1][i].bar(d['hist'][1][:-1], d['hist'][0], width=1, color='gray')
                axs[0][i].set_title('{}:\n{:.2f}%'.format(self._split_long_titles(d['name']), 
                                                          d['avg_effect']))
            else:
                axs[0][i].plot(d['x'], d['y'], c='gray')
                axs[1][i].bar(d['hist'][1][:-1], d['hist'][0], width=1, color='gray')
                axs[0][i].set_title('{}:\n{:.2f}%'.format(self._split_long_titles(d['name']),
                                                          d['avg_effect']))
            if np.max(d['y']) > y_max:
                y_max = np.max(d['y'])
            if np.min(d['y']) < y_min:
                y_min = np.min(d['y'])

        for i in range(show_n):
            # fig.axes[i].set_ylim([y_min + 0.05 * y_min, y_max + 0.05 * y_max])
            axs[1][i].get_xaxis().set_visible(False)
            axs[1][i].get_yaxis().set_visible(False)
            # axs[1][i].set
        plt.show()

    def plot_interactions(self):
        for i, fp in enumerate(self.feat_pairs):
            x1 = np.linspace(self.unique[fp[0]].min(), self.unique[fp[0]].max(), 50)
            x2 = np.linspace(self.unique[fp[1]].min(), self.unique[fp[1]].max(), 50)
            pred = np.zeros((len(x1), len(x2)))
            for v in range(pred.shape[0]):
                x1_stat = x1[v] * np.ones(len(x2))
                for regressor, boost_rate in zip(self.regressors[self.single_regressors:], 
                                                 self.boosting_rates[self.single_regressors:]):
                    pred[v,:] += boost_rate * regressor.predict_single(x1_stat, x2, i).squeeze()
        
            fig, ax = plt.subplots()
            ax.pcolormesh(x1, x2, pred, shading='auto')
            #ax.set_title(f'{fp[0]} - {fp[1]}')
            plt.show()

    def plot_by_list(self, feat_names, plot=True):
        '''
        This function plots the shape functions of feat_names.
        '''
        feature_effects = self.get_shape_functions_as_dict()
        top_k = [d for d in sorted(feature_effects, reverse=True, key=lambda x: x['avg_effect'])]

        y_min = np.inf
        y_max = -np.inf
        fig, axs = plt.subplots(2, max(2, len(feat_names)), figsize=(14, 4), 
                                gridspec_kw={'height_ratios': [5, 1]})
        i = 0
        for d in top_k:
            if d['name'] not in feat_names:
                continue
            if len(d['x']) < 4:
                axs[0][i].scatter(d['x'], d['y'], c='gray')
                axs[1][i].bar(d['hist'][1][:-1], d['hist'][0],width=1, color='gray')
                axs[0][i].set_title('{}:\n{:.2f}%'.format(self._split_long_titles(d['name']), 
                                                         d['avg_effect']))
            else:
                axs[0][i].plot(d['x'], d['y'], c='gray')
                axs[1][i].bar(d['hist'][1][:-1], d['hist'][0],width=1, color='gray')
                axs[0][i].set_title('{}:\n{:.2f}%'.format(self._split_long_titles(d['name']),
                                                         d['avg_effect']))
            if np.max(d['y']) > y_max:
                y_max = np.max(d['y'])
            if np.min(d['y']) < y_min:
                y_min = np.min(d['y'])
            i += 1

        for i in range(len(feat_names)):
            # fig.axes[i].set_ylim([y_min+0.05*y_min, y_max+0.05*y_max])
            axs[1][i].get_xaxis().set_visible(False)
            axs[1][i].get_yaxis().set_visible(False)
            #axs[1][i].set
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
    from sklearn.datasets import make_circles
    import seaborn as sns
    
    X_small, y_small = make_circles(n_samples=(250,500), random_state=3, noise=0.04, factor = 0.3)
    X_large, y_large = make_circles(n_samples=(250,500), random_state=3, noise=0.04, factor = 0.7)
    
    y_small[y_small==1] = 0
    
    df = pd.DataFrame(np.vstack([X_small,X_large]),columns=['x1','x2'])
    df['label'] = np.hstack([y_small,y_large])
    df.label = 2 * df.label - 1
    
    sns.scatterplot(data=df,x='x1',y='x2',hue='label')
    
    m = IGANN(n_estimators=1000, n_hid=7, only_interactions=True, interactions=1, verbose=2)
    m.fit(df[['x1', 'x2']], df.label)
