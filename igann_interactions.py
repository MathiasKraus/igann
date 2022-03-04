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
    def __init__(self, n_input, n_hid, feat_pairs, seed=0, scale=10, scale_inter=1, elm_alpha=0.0001, act='elu'):
        '''
        Input parameters:
        - n_input: number of inputs/features for your task (shape[1] of your X matrix)
        - n_hid: number of hidden neurons for each feature. Multiply this number with n_input
        to get the number of parameters this model has (-1 because of the bias term)
        - feat_pairs: TODO
        - seed: This number sets the seed for generating the random weights. It should
        be different for each regressor
        - scale: the scale which is used to initialize the weights in the first layer of the 
        model. These weights are not changed throughout the optimization. This parameter
        has huge impact on the model. A larger scale makes the shape functions sharper, a lower
        scale makes them smoother. Definitely play around with this one.
        - scale_inter: scale of the interaction ELMs
        - elm_alpha: the regularization of the ridge regression. I didn't find this to have
        a huge impact, but let's try it out...
        - act: the activation function in the model. I mainly used 'elu' which works well,
        'relu' is kind of a mess, 'gelu' also provides nice results.
        '''
        super().__init__()
        np.random.seed(seed)
        # The following are the random weights in the model which are not optimized. 
        self.hidden_list = [np.random.normal(scale=scale, size=(1, n_hid)) for _ in range(n_input)]
        self.hidden_list_inter = [(np.random.normal(scale=scale_inter, size=(1, n_hid)),
                                  np.random.normal(scale=scale_inter, size=(1, n_hid))) for _ in range(len(feat_pairs))]
        self.output_model = None
        self.n_input = n_input
        self.n_hid = n_hid
        self.feat_pairs = feat_pairs
        self.scale = scale
        self.scale_inter = scale_inter
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
        X_hid = np.zeros((X.shape[0], (X.shape[1] + len(self.feat_pairs)) * self.n_hid))
        for i in range(X.shape[1]):
            x_in = np.expand_dims(X[:, i], 1)
            x_in = np.dot(x_in, self.hidden_list[i])
            x_in = self.act(x_in)
            X_hid[:, i * self.n_hid:(i + 1) * self.n_hid] = x_in
        
        starting_index = X.shape[1] * self.n_hid
            
        for c, (i, j) in enumerate(self.feat_pairs):
            x_in = np.expand_dims(X[:, i], 1)
            x_in = np.dot(x_in, self.hidden_list_inter[c][0])
            x_jn = np.expand_dims(X[:, j], 1)
            x_jn = np.dot(x_jn, self.hidden_list_inter[c][1])
            
            X_hid[:, starting_index + c * self.n_hid : starting_index + (c + 1) * self.n_hid] = self.act(x_in + x_jn)
        
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
    
    def predict_single_inter(self, x1, x2, i):
        '''
        This function computes the output of one shape function. Note, that the 
        bias term is not used for this prediction.
        Input parameters:
        x: a vector representing the values which are used for feature i
        i: the index of the feature that should be used for the prediction
        '''

        x1 = x1.reshape(len(x1), 1)
        x2 = x2.reshape(len(x2), 1)
        
        x1 = np.dot(x1, self.hidden_list_inter[i][0])
        x2 = np.dot(x2, self.hidden_list_inter[i][1])
        x = self.act(x1 + x2)
        
        starting_index = self.n_input * self.n_hid
        out = np.dot(x, np.expand_dims(self.output_model.coef_[starting_index + i * self.n_hid: 
                                                               starting_index + (i + 1) * self.n_hid], 1))
        return out

    def fit(self, X, y, mult_coef):
        '''
        This function fits the ELM on the training data (X, y). 
        '''
        X_hid = self.get_hidden_values(X)*mult_coef

        # Fit the ridge regression on the hidden values.
        m = Ridge(alpha=self.elm_alpha, tol=0.01, fit_intercept=False)
        # TODO: Intercept?
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
    def __init__(self, task='classification', n_hid=10, n_estimators=5000, boost_rate=0.1, init_reg=1.0, 
    		     elm_scale=1, elm_scale_inter=0.5, elm_alpha=1, feat_select=None, interactions=0, 
                 act='elu', early_stopping=50, random_state=1,
                 verbose=1):
        '''
        Initialize the model. Input parameters:
        task: defines the task, can be 'regression' or 'classification'
        n_hid: the number of hidden neurons for one feature
        n_estimators: the maximum number of estimators (ELMs) to be fitted.
        boost_rate_opt: Flag that indicates if the boosting rates are fine-tuned or not.
        boost_rate_opt_iter: Number of iterations after which the fine-tuning step of the boosting rates is performed
        It can be a real number 0.5/1.0,... or 'auto'. With 'auto' the boosting_rate is optimized.
        init_reg: the initial regularization strength for the linear model.
        elm_scale: the scale of the random weights in the elm model.
        elm_scale_inter: the scale of the random weights for the interaction terms.
        elm_alpha: the regularization strength for the ridge regression in the ELM model.
        feat_select: Integer that says how many features should be selected
        interactions: the number of interactions that should be fit.
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
        self.elm_scale = elm_scale
        self.elm_scale_inter = elm_scale_inter
        self.elm_alpha = elm_alpha
        self.init_reg = init_reg
        self.act = act
        self.n_estimators = n_estimators
        self.early_stopping = early_stopping
        self.feat_select = feat_select
        self.interactions = interactions
        self.random_state = random_state
        self.verbose = verbose
        self.top_k_features = []
        self.regressors = []
        self.boosting_rates = []
        self.train_scores = []
        self.val_scores = []
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.regressor_predictions = []
        self.boost_rate = boost_rate

        if task == 'classification':
            self.init_classifier = LogisticRegression(penalty='l1', solver='liblinear', C=self.init_reg,
                                                      random_state=random_state)
        elif task == 'regression':
            self.init_classifier = Lasso(alpha=1/self.init_reg)
        else:
            print('Task not implemented. Can be classification or regression')
    
    def _loss_sqrt_hessian(self, y, p):
        '''
        This function computes the square root of the hessians of the log loss or the mean squared error.
        '''
        if self.task == 'classification':
            return 0.5 / np.cosh(0.5*y*p)
        else:
            return np.array([np.sqrt(2.0)])

    def _get_y_tilde(self,y,p):
        if self.task == 'classification':
            return y/np.exp(0.5*y*p)
        else:
            return np.sqrt(2.0)*(y - p)
    
    def _act(self, x):
        x[x <= 0] = np.exp(x[x <= 0]) - 1
        return x

    def fit(self, X, y, feat_pairs=None, fixed_feat=[],
            val_set=None, eval=None, plot_fixed_features=None):
        '''
        This function fits the model on training data (X, y). 
        Parameters:
        X: the feature matrix
        y: the targets
        feat_pairs: Given feature pairs
        fixed_feat: List of feature names which should definitely end up in the model
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
            if np.min(y) != -1:
                y = 2 * y - 1

        if feat_pairs != None:
            self.interactions = len(feat_pairs)

        if self.feat_select != None and feat_pairs != None:
            for fp in feat_pairs:
                if fp[0] not in fixed_feat:
                    fixed_feat.append(fp[0])
                if fp[1] not in fixed_feat:
                    fixed_feat.append(fp[1])

        if self.feat_select != None:            
            feature_indizes = self._select_features(X, y)
            feature_indizes.extend([e for e,f in enumerate(self.feature_names) if f in fixed_feat])
            self.feature_names = np.array([f for e, f in enumerate(self.feature_names) if e in feature_indizes])
            X = X[:, feature_indizes]
            self.feature_indizes = feature_indizes
        else:
            self.feature_indizes = np.arange(X.shape[1])

        # Fit the linear model on all data
        self.init_classifier.fit(X, y)

        #print(self.init_classifier.coef_)

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

        if self.interactions==0:
            self.feat_pairs = []
        else:
            hessian_train_sqrt=self._loss_sqrt_hessian(y, y_hat)
            y_tilde=self._get_y_tilde(y,y_hat)
            
            if feat_pairs != None:
                self.feat_pairs = []
                if type(feat_pairs[0][0]) == str:
                    for fp in feat_pairs:
                        fp_comb = (int(np.where(self.feature_names == fp[0])[0]), 
                                   int(np.where(self.feature_names == fp[1])[0]))
                        self.feat_pairs.append(fp_comb)
                else:
                    self.feat_pairs = feat_pairs
            else:
                self.feat_pairs = self._find_interactions(X, y_tilde, 1/np.sqrt(0.5)*hessian_train_sqrt[:,np.newaxis])
            

        self._run_optimization(X, y, y_hat, X_val, y_val, y_hat_val, eval,
                               val_loss_init, plot_fixed_features,
                               feat_pairs=self.feat_pairs)

        return 

    def _run_optimization(self, X, y, y_hat, X_val, y_val, y_hat_val, eval, 
                         best_loss, plot_fixed_features, 
                         feat_pairs=[]):
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
        feat_pairs: list of feature pairs when fitting interactions
        '''

        counter_no_progress = 0
        best_iter = 0
        # Sequentially fit one ELM after the other. Max number is stored in self.n_estimators.
        for counter in range(self.n_estimators):
            
            hessian_train_sqrt=self._loss_sqrt_hessian(y, y_hat)
            y_tilde=np.sqrt(0.5)*self._get_y_tilde(y,y_hat)

            # Fit an ELM on y_tilde_tilde
            regressor = ELM_Regressor(n_input=X.shape[1], n_hid=self.n_hid, 
                                      feat_pairs=feat_pairs, seed=counter, scale=self.elm_scale, scale_inter=self.elm_scale_inter, elm_alpha=self.elm_alpha,
                                      act=self.act)
            
            # Fit  ELM regressor
            regressor.fit(X, y_tilde, np.sqrt(0.5)*self.boost_rate*hessian_train_sqrt[:,np.newaxis])

            # Make a prediction of the ELM for the gradients of train and val
            train_regressor_pred = regressor.predict(X).squeeze()
            val_regressor_pred = regressor.predict(X_val).squeeze()

            self.regressor_predictions.append(train_regressor_pred)

            # Update the prediction for training and validation data
            y_hat += self.boost_rate * train_regressor_pred
            y_hat_val += self.boost_rate * val_regressor_pred
            
            if self.task == 'classification':
                val_pred_prob = 1 / (1 + np.exp(-y_hat_val))
                train_pred_prob = 1 / (1 + np.exp(-y_hat))

                val_auc = roc_auc_score(y_val, val_pred_prob)
                val_loss = log_loss(y_val, val_pred_prob)

                train_auc = roc_auc_score(y, train_pred_prob)
                train_loss = log_loss(y, train_pred_prob)
            else:
                val_mse = mean_squared_error(y_val, y_hat_val)
                val_loss = val_mse
                train_mse = mean_squared_error(y, y_hat)
                train_loss = train_mse

            # Keep the ELM, the boosting rate and losses in lists, so 
            # we can later use them again.
            self.regressors.append(regressor)
            self.boosting_rates.append(self.boost_rate)
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
                    self._print_results(counter, counter_no_progress, eval, self.boost_rate, train_auc,
                                       train_loss, val_auc, val_loss)
                else:
                    self._print_results(counter, counter_no_progress, eval, self.boost_rate, train_mse,
                                       train_loss, val_mse, val_loss)
                
            # Stop training if the counter for early stopping is greater than the parameter we passed.
            if counter_no_progress > self.early_stopping and self.early_stopping > 0:
                break

            if self.verbose >= 2:
                if counter%5==0:
                    if plot_fixed_features != None:
                        self.plot_single(plot_by_list=plot_fixed_features)    
                    else:
                        self.plot_single()
                        if len(self.feat_pairs) > 0:
                            if counter==0:
                                self.plot_interactions(True)
                            else:
                                self.plot_interactions(False)
            
        if self.early_stopping > 0:
            # We remove the ELMs that did not improve the performance. Most likely best_iter equals self.early_stopping.
            if self.verbose > 0:
                print(f'Cutting at {best_iter}') 
            self.regressors = self.regressors[:best_iter]
            self.boosting_rates = self.boosting_rates[:best_iter]
            
        return best_loss

    def _select_features(self, X, y):
        regressor = ELM_Regressor(X.shape[1], self.n_hid, [],
                                  scale=self.elm_scale, scale_inter=self.elm_scale_inter, act=self.act)
        X_tilde = regressor.get_hidden_values(X)

        lower_bound = 1e-10
        upper_bound = 1000
        alpha = (lower_bound+upper_bound)/2
        found = False
        for _ in range(1000):
            lasso = Lasso(alpha=alpha, random_state=self.random_state)
            lasso.fit(X_tilde, y)
            features = []
            for pos in np.where(lasso.coef_ != 0)[0]:
                block = int((pos - (pos % self.n_hid)) / self.n_hid)
                if block not in features:
                    features.append(block)
                
            if len(features) == self.feat_select:
                found = True
                break
            elif len(features) < self.feat_select:
                upper_bound=alpha
                alpha = (lower_bound+upper_bound)/2
            else:
                lower_bound=alpha
                alpha = (lower_bound+upper_bound)/2
      
        if not found:
            print('Did not find wanted number of features!!!!')
            print(f'Using {features}')
        
        else:
            if self.verbose > 0:
                print(f'Found features {features}')
            
        return features
        

    def _find_interactions(self, X, y, mult_coef):
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
            if self.task == 'classification':
                mult_coef = mult_coef[sample]
        
        regressor = ELM_Regressor(X.shape[1], self.n_hid, [(i,j) for i in range(X.shape[1]) for j in range(X.shape[1])],
                                  scale=self.elm_scale, scale_inter=self.elm_scale_inter, act=self.act)
        X_tilde = regressor.get_hidden_values(X)
        X_tilde_tilde = X_tilde*mult_coef

        lower_bound = 1e-10
        upper_bound = 1000
        alpha = (lower_bound+upper_bound)/2
        found = False
        for _ in range(1000):
            lasso = Lasso(alpha=alpha, random_state=self.random_state)
            lasso.fit(X_tilde_tilde, y)
            feat_pairs = self._get_nonzero_feat_pairs(lasso.coef_[self.n_hid*X.shape[1]:], X)
            if len(feat_pairs) == self.interactions:
                found = True
                break
            elif len(feat_pairs) < self.interactions:
                upper_bound=alpha
                alpha = (lower_bound+upper_bound)/2
            else:
                lower_bound=alpha
                alpha = (lower_bound+upper_bound)/2
      
        if not found:
            print('Did not find wanted number of interactions!!!!')
            print(f'Using interactions {feat_pairs}')
        
        else:
            if self.verbose > 0:
                print(f'Found interactions {feat_pairs}')
            
        return feat_pairs
        

    def _get_nonzero_feat_pairs(self, coef, X):
        '''
        This function computes how many ELM blocks are corresponding to coefficients != 0.
        It then returns all feature pairs that correspond to these ELM blocks.
        '''
        feat_pairs = []
        for pos in np.where(coef != 0)[0]:
            block = int((pos - (pos % self.n_hid)) / self.n_hid)
            corr_i_feat = int(block / X.shape[1])
            corr_j_feat = block % X.shape[1]
            if corr_i_feat == corr_j_feat:
                continue
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

    def _update_predict(self, X, pred, boost_rate, regressor):
        '''
        This is a helper function that speeds up training when we pass an evaluation set.
        '''
        if type(X) == pd.DataFrame:
            X = np.array(X)
        pred_nn = boost_rate * regressor.predict(X).squeeze()
        pred += pred_nn

        return pred

    def predict(self, X):
        '''
        This function returns a prediction for a given feature matrix X.
        Note: for a classification task, this simply returns the raw logit values.
        '''
        if type(X) == pd.DataFrame:
            X = np.array(X)
        X = X[:, self.feature_indizes]
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
            linspac = self.unique[i]
            if self.task == 'classification':
                pred = self.init_classifier.coef_[0, i] * linspac
            else:
                pred = self.init_classifier.coef_[i] * linspac
            for regressor, boost_rate in zip(self.regressors, self.boosting_rates):
                pred += (boost_rate * regressor.predict_single(linspac.reshape(-1, 1), i).squeeze())
            feature_effects.append(
                {'name': feat_name, 'x': linspac, 
                    'y': pred, 'avg_effect': np.mean(np.abs(pred)),
                    'hist': self.hist[i]})

        overall_effect = np.sum([d['avg_effect'] for d in feature_effects])
        for d in feature_effects:
            d['avg_effect'] = d['avg_effect'] / overall_effect * 100
            
        return feature_effects

    def plot_single(self, plot_by_list=None, show_n=5):
        '''
        This function plots the most important shape functions. 
        Parameters:
        show_n: the number of shape functions that should be plotted (don't know if this works).
        '''
        feature_effects = self.get_shape_functions_as_dict()
        if plot_by_list is None:
            top_k = [d for d in sorted(feature_effects, reverse=True, key=lambda x: x['avg_effect'])][:show_n]
        else:
            top_k = [d for d in sorted(feature_effects, reverse=True, key=lambda x: x['avg_effect'])]
            show_n = len(plot_by_list)

        if [d['name'] for d in top_k] != self.top_k_features:
            create_figure = True
            self.top_k_features = plot_by_list
        else:
            create_figure=False

        if create_figure:
            plt.close(fig="Shape functions")
            self.fig, self.axs = plt.subplots(2, show_n, figsize=(14, 4),
                                    gridspec_kw={'height_ratios': [5, 1]},
                                    num="Shape functions")
            plt.subplots_adjust(wspace=0.4)
            self.plot_objects=[]
        else:
            plot_object_counter = 0
            
        i = 0
        for d in top_k:
            if plot_by_list is not None and d['name'] not in plot_by_list:
                continue
            if len(d['x']) < 4:
                if create_figure:
                    if show_n == 1:
                        plot_object = self.axs[0].bar(d['x'], d['y'], color='black', 
                                                      width = 0.1*(d['x'].max() - d['x'].min()))
                        self.axs[1].bar(d['hist'][1][:-1], d['hist'][0], width=1, color='gray')
                        self.axs[0].set_title('{}:\n{:.2f}%'.format(self._split_long_titles(d['name']), 
                                                                    d['avg_effect']))
                    else:
                        plot_object = self.axs[0][i].bar(d['x'], d['y'], color='black', 
                                                      width = 0.1*(d['x'].max() - d['x'].min()))
                        self.axs[1][i].bar(d['hist'][1][:-1], d['hist'][0], width=1, color='gray')
                        self.axs[0][i].set_title('{}:\n{:.2f}%'.format(self._split_long_titles(d['name']), 
                                                                       d['avg_effect']))
                    if type(plot_object)==list:
                        plot_object=plot_object[0]
                    
                    self.plot_objects.append(plot_object)
                else:
                    plot_object = self.plot_objects[plot_object_counter]
                    plot_object_counter += 1
                    plot_object.set_data(d['x'], d['y'])
                    self.axs[0][i].set_ylim(bottom=min(d['y']), top=max(d['y']))
                    self.fig.canvas.draw_idle()
                    self.fig.canvas.flush_events()
            else:
                if create_figure:
                    if show_n == 1:
                        plot_object = self.axs[0].plot(d['x'], d['y'], c='black')
                        self.axs[1].bar(d['hist'][1][:-1], d['hist'][0], width=1, color='gray')
                        self.axs[0].set_title('{}:\n{:.2f}%'.format(self._split_long_titles(d['name']),
                                                              d['avg_effect']))
                    else:
                        plot_object = self.axs[0][i].plot(d['x'], d['y'], c='black')
                        self.axs[1][i].bar(d['hist'][1][:-1], d['hist'][0], width=1, color='gray')
                        self.axs[0][i].set_title('{}:\n{:.2f}%'.format(self._split_long_titles(d['name']),
                                                                       d['avg_effect']))
                    if type(plot_object)==list:
                        plot_object=plot_object[0]
                        
                    self.plot_objects.append(plot_object)
                else:
                    plot_object = self.plot_objects[plot_object_counter]
                    plot_object_counter += 1
                    plot_object.set_data(d['x'],d['y'])
                    self.axs[0][i].set_ylim(bottom=min(d['y']), top=max(d['y']))
                    self.fig.canvas.draw_idle()
                    self.fig.canvas.flush_events()
            i += 1
            
        if create_figure:   
            if show_n == 1:
                self.axs[1].get_xaxis().set_visible(False)
                self.axs[1].get_yaxis().set_visible(False)
            else:
                for i in range(show_n):
                    self.axs[1][i].get_xaxis().set_visible(False)
                    self.axs[1][i].get_yaxis().set_visible(False)
            plt.show()

    def plot_interactions_categorical_continuous(self, cat_feat, cont_feat, feat_pair_num):
        #plt.close(fig="Interactions")
        #self.fig_inter, self.axs_inter = plt.subplots(1, 1, figsize=(14, 10), num="Interactions")
        #plt.subplots_adjust(wspace=0.4)

        for x1 in self.unique[cat_feat]:
            x2 = np.linspace(self.unique[cont_feat].min(), self.unique[cont_feat].max(), 200)
            pred = np.zeros(len(x2))
            x1_stat = x1 * np.ones(len(x2))

            for regressor, boost_rate in zip(self.regressors, 
                                                self.boosting_rates):
                if cat_feat < cont_feat:
                    pred += boost_rate * regressor.predict_single_inter(x1_stat, x2, feat_pair_num).squeeze()
                else:
                    pred += boost_rate * regressor.predict_single_inter(x2, x1_stat, feat_pair_num).squeeze()
        
            if self.task == 'classification':
                single_pred_x1 = self.init_classifier.coef_[0, cat_feat] * x1
                single_pred_x2 = self.init_classifier.coef_[0, cont_feat] * x2
            else:
                single_pred_x1 = self.init_classifier.coef_[cat_feat] * x1
                single_pred_x2 = self.init_classifier.coef_[cont_feat] * x2

            for regressor, boost_rate in zip(self.regressors, self.boosting_rates):
                single_pred_x1 += (boost_rate * regressor.predict_single(x1.reshape(-1, 1), cat_feat).squeeze())
                single_pred_x2 += (boost_rate * regressor.predict_single(x2.reshape(-1, 1), cont_feat).squeeze())
            
            
            pred += single_pred_x2
            pred += single_pred_x1

            plt.plot(x2, pred, label=str(x1))
        
        plt.legend()
        plt.show()
            

    def plot_interactions(self, create_figure=True):
        if create_figure:
            plt.close(fig="Interactions")
            self.fig_inter, self.axs_inter = plt.subplots(1, len(self.feat_pairs), figsize=(14, 10), num="Interactions")
            plt.subplots_adjust(wspace=0.4)
            self.plot_objects_inter=[]
        else:
            plot_object_counter = 0
        
        for i, fp in enumerate(self.feat_pairs):
            x1 = np.linspace(self.unique[fp[0]].min(), self.unique[fp[0]].max(), 50)
            x2 = np.linspace(self.unique[fp[1]].min(), self.unique[fp[1]].max(), 50)
            pred = np.zeros((len(x1), len(x2)))
            for v in range(pred.shape[0]):
                x1_stat = x1[v] * np.ones(len(x2))
                for regressor, boost_rate in zip(self.regressors, 
                                                 self.boosting_rates):
                    pred[v,:] += boost_rate * regressor.predict_single_inter(x1_stat, x2, i).squeeze()
            
            if create_figure:
                if len(self.feat_pairs)==1:
                    plot_object = self.axs_inter.pcolormesh(x1, x2, pred, shading='nearest')
                    #self.axs_inter.set_title('Interaction ({},{})'.format(self.feature_names[self.feat_pairs[0][0]],
                    #                                                      self.feature_names[self.feat_pairs[0][1]]))
                    self.axs_inter.set_title('Min: {:.2f}, Max: {:.2f})'.format(np.min(pred), np.max(pred)))
                    self.axs_inter.set_xlabel(self.feature_names[self.feat_pairs[0][0]])
                    self.axs_inter.set_ylabel(self.feature_names[self.feat_pairs[0][1]])
                    
                    self.axs_inter.set_aspect('equal', 'box')
                    self.plot_objects_inter.append(plot_object)
                    plt.show()
                else:
                    plot_object = self.axs_inter[i].pcolormesh(x1, x2, pred, shading='nearest')
                    #self.axs_inter[i].set_title('Interaction ({},{})'.format(self.feature_names[self.feat_pairs[i][0]],
                    #                                                      self.feature_names[self.feat_pairs[i][1]]))
                    self.axs_inter[i].set_title('Min: {:.2f}, Max: {:.2f})'.format(np.min(pred), np.max(pred)))
                    self.axs_inter[i].set_xlabel(self.feature_names[self.feat_pairs[i][0]])
                    self.axs_inter[i].set_ylabel(self.feature_names[self.feat_pairs[i][1]])
                    
                    self.axs_inter[i].set_aspect('equal', 'box')
                    self.plot_objects_inter.append(plot_object)
                    
            else:
                plot_object = self.plot_objects_inter[plot_object_counter]
                plot_object_counter +=1
                plot_object.set_array(pred)
                self.fig_inter.canvas.draw()
                self.fig_inter.canvas.flush_events()
        plt.show()

    def plot_interactions_plus_single(self, create_figure=True):
        if create_figure:
            plt.close(fig="Interactions_plus_single")
            self.fig_inter, self.axs_inter = plt.subplots(1, len(self.feat_pairs), figsize=(14, 4), num="Interactions_plus_single")
            plt.subplots_adjust(wspace=0.4)
            self.plot_objects_inter=[]
        else:
            plot_object_counter = 0
        
        for i, fp in enumerate(self.feat_pairs):
            x1 = np.linspace(self.unique[fp[0]].min(), self.unique[fp[0]].max(), 50)
            x2 = np.linspace(self.unique[fp[1]].min(), self.unique[fp[1]].max(), 50)
            pred = np.zeros((len(x1), len(x2)))
            for v in range(pred.shape[0]):
                x1_stat = x1[v] * np.ones(len(x2))
                for regressor, boost_rate in zip(self.regressors, 
                                                 self.boosting_rates):
                    pred[v,:] += boost_rate * regressor.predict_single_inter(x1_stat, x2, i).squeeze()
            
            if self.task == 'classification':
                single_pred_x1 = self.init_classifier.coef_[0, fp[0]] * x1
                single_pred_x2 = self.init_classifier.coef_[0, fp[1]] * x2
            else:
                single_pred_x1 = self.init_classifier.coef_[fp[0]] * x1
                single_pred_x2 = self.init_classifier.coef_[fp[1]] * x2
            # print(pred)
            for regressor, boost_rate in zip(self.regressors, self.boosting_rates):
                single_pred_x1 += (boost_rate * regressor.predict_single(x1.reshape(-1, 1), fp[0]).squeeze())
                single_pred_x2 += (boost_rate * regressor.predict_single(x2.reshape(-1, 1), fp[1]).squeeze())
            
            for j in range(pred.shape[0]):
                pred[j,:] += single_pred_x2
            
            for j in range(pred.shape[1]):
                pred[:,j] += single_pred_x1
            
            if create_figure:
                if len(self.feat_pairs)==1:
                    plot_object = self.axs_inter.pcolormesh(x1, x2, pred, shading='nearest')
                    self.axs_inter.set_title('Min: {:.2f}, Max: {:.2f})'.format(np.min(pred), np.max(pred)))
                    self.axs_inter.set_xlabel(self.feature_names[self.feat_pairs[0][0]])
                    self.axs_inter.set_ylabel(self.feature_names[self.feat_pairs[0][1]])
                    
                    self.axs_inter.set_aspect('equal', 'box')
                    self.plot_objects_inter.append(plot_object)
                    plt.show()
                else:
                    plot_object = self.axs_inter[i].pcolormesh(x1, x2, pred, shading='nearest')
                    self.axs_inter[i].set_title('Min: {:.2f}, Max: {:.2f})'.format(np.min(pred), np.max(pred)))
                    self.axs_inter[i].set_xlabel(self.feature_names[self.feat_pairs[i][0]])
                    self.axs_inter[i].set_ylabel(self.feature_names[self.feat_pairs[i][1]])
                    
                    self.axs_inter[i].set_aspect('equal', 'box')
                    self.plot_objects_inter.append(plot_object)
                    plt.show()
            else:
                plot_object = self.plot_objects_inter[plot_object_counter]
                plot_object_counter +=1
                plot_object.set_array(pred)
                self.fig_inter.canvas.draw()
                self.fig_inter.canvas.flush_events()


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
    inputs = np.random.random((1000, 10))
    inputs[:100, -1] *= -1
    targets = np.random.random((1000, 1))
    targets[:100] *= 2

    print(np.mean(targets[:100]))
    print(np.mean(targets[100:]))

    binary_groups = np.array([0] * 100 + [1] * 900, dtype=np.int32)
    
    m = IGANN(task='regression', n_estimators=3, n_hid=5, boost_rate=0.3, interactions=0, verbose=2)
    m.fit(inputs, targets.squeeze())
    
    aaa
    
    
    from sklearn.datasets import make_circles
    import seaborn as sns
    
    X_small, y_small = make_circles(n_samples=(250,500), random_state=3, noise=0.04, factor = 0.3)
    X_large, y_large = make_circles(n_samples=(250,500), random_state=3, noise=0.04, factor = 0.7)
    
    y_small[y_small==1] = 0
    
    df = pd.DataFrame(np.vstack([X_small,X_large]),columns=['x1','x2'])
    df['label'] = np.hstack([y_small,y_large])
    df.label = 2 * df.label - 1
    
    sns.scatterplot(data=df,x='x1',y='x2',hue='label')
    
    m = IGANN(n_estimators=50000, n_hid=10, elm_alpha=5, boost_rate=1, interactions=1, verbose=2)
    m.fit(df[['x1', 'x2']], df.label)