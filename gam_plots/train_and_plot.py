import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from interpret import show
from pygam import LogisticGAM, LinearGAM
from gaminet import GAMINet
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, RobustScaler
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from load_datasets import load_adult_data, load_crimes_data
import tensorflow as tf
import torch


random_state = 1
task = 'regression'  # regression or classification

dataset = load_crimes_data()
dataset_name = 'crimes'

X = pd.DataFrame(dataset['full']['X'])
y = np.array(dataset['full']['y'])
X, y = shuffle(X, y, random_state=random_state)

is_cat = np.array([dt.kind == 'O' for dt in X.dtypes])

num_cols = X.columns.values[~is_cat]
# one hot encoder pipeline replaced by pd.getdummies to make sure column names are concatenated
# cat_ohe_step = ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))

# Handle unknown data as ignore
X = pd.get_dummies(X)
X = X.reindex(columns=X.columns, fill_value=0)
dummy_column_names = X.columns

# cat_pipe = Pipeline([cat_ohe_step])
num_pipe = Pipeline([('identity', FunctionTransformer())])  # , ('scaler', RobustScaler())])
transformers = [
    # ('cat', cat_pipe, cat_cols) replaced by pd.getdummies
    ('num', num_pipe, num_cols)
]
ct = ColumnTransformer(transformers=transformers, remainder='passthrough')
ct.fit(X)
X = ct.transform(X)

X = pd.DataFrame(X, columns=dummy_column_names)

scaler_dict = {}
for c in num_cols:

    scaler = MinMaxScaler((0, 1))
    scaler.fit([[0], [1]])
    X[c] = scaler.transform(X[c].values.reshape(-1, 1))
    # scaler = RobustScaler()
    # X[c] = scaler.fit_transform(X[c].values.reshape(-1, 1))
    scaler_dict[c] = scaler


def feature_importance_visualize(data_dict_global, folder="./results/", name="demo", save_png=False, save_eps=False):
    all_ir = []
    all_names = []
    for key, item in data_dict_global.items():
        if item["importance"] > 0:
            all_ir.append(item["importance"])
            all_names.append(key)

    max_ids = len(all_names)
    if max_ids > 0:
        fig = plt.figure(figsize=(0.4 + 0.6 * max_ids, 4))
        ax = plt.axes()
        ax.bar(np.arange(len(all_ir)), [ir for ir, _ in sorted(zip(all_ir, all_names))][::-1])
        ax.set_xticks(np.arange(len(all_ir)))
        ax.set_xticklabels([name for _, name in sorted(zip(all_ir, all_names))][::-1], rotation=60)
        plt.xlabel("Feature Name", fontsize=12)
        plt.ylim(0, np.max(all_ir) + 0.05)
        plt.xlim(-1, len(all_names))
        plt.title("Feature Importance")

        save_path = folder + name
        if save_eps:
            if not os.path.exists(folder):
                os.makedirs(folder)
            fig.savefig("%s.eps" % save_path, bbox_inches="tight", dpi=100)
        if save_png:
            if not os.path.exists(folder):
                os.makedirs(folder)
            fig.savefig("%s.png" % save_path, bbox_inches="tight", dpi=100)
    plt.show()


# %%

def make_plot(x, mean, upper_bounds, lower_bounds, feature_name, model_name, dataset_name, num_epochs='', debug=False):
    x = np.array(x)
    if debug:
        print("Num cols:", num_cols)
    if feature_name in num_cols:
        if debug:
            print("Feature to scale back:", feature_name)
        # todo: x = scaler_dict[feature_name].inverse_transform(x.reshape(-1, 1)).squeeze()
    else:
        if debug:
            print("Feature not to scale back:", feature_name)

    plt.plot(x, mean, color='black')
    plt.fill_between(x, lower_bounds, mean, color='gray')
    plt.fill_between(x, mean, upper_bounds, color='gray')
    plt.xlabel(f'Feature value')
    plt.ylabel('Feature effect on model output')
    plt.title(f'Feature:{feature_name}')
    plt.savefig(f'plots/{model_name}_{dataset_name}_shape_{feature_name}_{num_epochs}epochs.pdf')
    plt.show()


def make_plot_interaction(left_names, right_names, scores, feature_name_left, feature_name_right, model_name,
                          dataset_name):
    left_names = np.array(left_names)
    # print(right_names)
    # if feature_name_left in num_cols:
        # todo:  left_names = scaler_dict[feature_name_left].inverse_transform(left_names.reshape(-1, 1)).squeeze()
    right_names = np.array(right_names)
    # if feature_name_right in num_cols:
        # todo:  right_names = scaler_dict[feature_name_right].inverse_transform(right_names.reshape(-1, 1)).squeeze()
    fig, ax = plt.subplots()
    im = ax.pcolormesh(left_names, right_names, scores, shading='auto')
    fig.colorbar(im, ax=ax)
    plt.xlabel(feature_name_left)
    plt.ylabel(feature_name_right)
    plt.savefig(f'plots/{model_name}_{dataset_name}_interact_{feature_name_left}x{feature_name_right}.png')
    plt.show()


def make_one_hot_plot(class_zero, class_one, feature_name, model_name, dataset_name, num_epochs=''):
    plt.bar([0, 1], [class_zero, class_one], color='gray', tick_label=[f'{feature_name} = 0', f'{feature_name} = 1'])
    plt.ylabel('Feature effect on model output')
    plt.title(f'Feature:{feature_name}')
    plt.savefig(f'plots/{model_name}_{dataset_name}_onehot_{feature_name}_{num_epochs}epochs.pdf')
    plt.show()


# %%

def EBM_show(X, y):
    m4 = ExplainableBoostingRegressor(interactions=10, max_bins=256)
    m4.fit(X, y)
    ebm_global = m4.explain_global()
    show(ebm_global)


def EBM(X, y, dataset_name, model_name='EBM'):
    if task == "classification":
        m4 = ExplainableBoostingClassifier(interactions=10, max_bins=1000)
    else:
        m4 = ExplainableBoostingRegressor(interactions=10, max_bins=1000)
    m4.fit(X, y)
    ebm_global = m4.explain_global()
    for i in range(len(ebm_global.data()['names'])):
        data_names = ebm_global.data()
        feature_name = data_names['names'][i]
        shape_data = ebm_global.data(i)
        if shape_data['type'] == 'interaction':
            x_name, y_name = feature_name.split('x')
            x_name = x_name.replace(' ', '')
            y_name = y_name.replace(' ', '')
            make_plot_interaction(shape_data['left_names'], shape_data['right_names'],
                                  np.transpose(shape_data['scores']),
                                  x_name, y_name, model_name, dataset_name)
            continue
        if len(shape_data['names']) == 2:
            make_one_hot_plot(shape_data['scores'][0], shape_data['scores'][1], feature_name, model_name, dataset_name)
        else:
            make_plot(shape_data['names'][:-1], shape_data['scores'], shape_data['scores'],
                      shape_data['scores'], feature_name, model_name, dataset_name)

    feat_for_vis = dict()
    for i, n in enumerate(ebm_global.data()['names']):
        feat_for_vis[n] = {'importance': ebm_global.data()['scores'][i]}
    feature_importance_visualize(feat_for_vis, save_png=True, folder='.', name='ebm_feat_imp')

def GAM(X, y, dataset_name, model_name='GAM'):
    if task == "classification":
        gam = LogisticGAM().fit(X, y)
    elif task == "regression":
        gam = LinearGAM().fit(X, y)
    for i, term in enumerate(gam.terms):
        if term.isintercept:
            continue
        XX = gam.generate_X_grid(term=i)
        pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
        # make_plot(XX[:,i].squeeze(), pdep, confi[:,0], confi[:,1], X.columns[i])
        if len(X[X.columns[i]].unique()) == 2:
            make_one_hot_plot(pdep[0], pdep[-1], X.columns[i], model_name, dataset_name)
        else:
            make_plot(XX[:, i].squeeze(), pdep, pdep, pdep, X.columns[i], model_name, dataset_name)

def Gaminet(X, y, dataset_name, model_name='Gaminet'):
    meta_info = {X.columns[i]: {'type': 'continuous'} for i in range(len(X.columns))}
    meta_info.update({'Y': {'type': 'target'}})

    # from sklearn.preprocessing import FunctionTransformer
    identity = FunctionTransformer()

    for i, (key, item) in enumerate(meta_info.items()):
        if item['type'] == 'target':
            continue
        # sx = MinMaxScaler((0, 1))
        # sx.fit([[0], [1]])
        # print(scaler_dict.keys())
        # print(X.columns)
        if key in scaler_dict:
            meta_info[key]['scaler'] = scaler_dict[key]
        else:
            meta_info[key]['scaler'] = identity

    if task == "classification":
        model_to_run = GAMINet(meta_info=meta_info, interact_num=20,
                               interact_arch=[40] * 5, subnet_arch=[40] * 5,
                               batch_size=1024, task_type="Classification", activation_func=tf.nn.relu,
                               main_effect_epochs=5, interaction_epochs=5, tuning_epochs=5,
                               # todo: main_effect_epochs=5000, interaction_epochs=5000, tuning_epochs=500,
                               lr_bp=[0.0001, 0.0001, 0.0001], early_stop_thres=[50, 50, 50],
                               heredity=True, loss_threshold=0.01, reg_clarity=1,
                               mono_increasing_list=[], mono_decreasing_list=[],  # the indices list of features
                               verbose=True, val_ratio=0.2, random_state=random_state)
        print(np.array(y).shape)
        model_to_run.fit(np.array(X), np.array(y).reshape(-1, 1))

    elif task == "regression":
        model_to_run = GAMINet(meta_info=meta_info, interact_num=20,
                               interact_arch=[40] * 5, subnet_arch=[40] * 5,
                               batch_size=1024, task_type="Regression", activation_func=tf.nn.relu,
                               main_effect_epochs=5000, interaction_epochs=5000, tuning_epochs=500,
                               lr_bp=[0.0001, 0.0001, 0.0001], early_stop_thres=[50, 50, 50],
                               heredity=True, loss_threshold=0.01, reg_clarity=1,
                               mono_increasing_list=[], mono_decreasing_list=[],  # the indices list of features
                               verbose=True, val_ratio=0.2, random_state=random_state)
        model_to_run.fit(np.array(X), np.array(y))

    data_dict = model_to_run.global_explain(save_dict=False, main_grid_size=1000)

    Xnames2Featurenames = dict(zip([X.columns[i] for i in range(X.shape[1])], X.columns))
    print(Xnames2Featurenames)

    for k in data_dict.keys():
        if data_dict[k]['type'] == 'pairwise':
            feature_name_left, feature_name_right = k.split('vs. ')
            feature_name_left = feature_name_left.replace(' ', '')
            feature_name_right = feature_name_right.replace(' ', '')
            make_plot_interaction(data_dict[k]['input1'], data_dict[k]['input2'], data_dict[k]['outputs'],
                                  Xnames2Featurenames[feature_name_left],
                                  Xnames2Featurenames[feature_name_right], model_name, dataset_name)
        elif data_dict[k]['type'] == 'continuous':
            # todo: reverse:
            # if len(X[Xnames2Featurenames[k]].unique()) == 2:
            #     make_one_hot_plot(data_dict[k]['outputs'][0], data_dict[k]['outputs'][-1],
            #                       Xnames2Featurenames[k], model_name, dataset_name)
            # else:
            try:
                make_plot(data_dict[k]['inputs'], data_dict[k]['outputs'], data_dict[k]['outputs'],
                          data_dict[k]['outputs'], Xnames2Featurenames[k], model_name, dataset_name)
            except:
                print("not continous")
        else:
            continue

    feat_for_vis = dict()
    for i, k in enumerate(data_dict.keys()):
        if 'vs.' in k:
            feature_name_left, feature_name_right = k.split('vs. ')
            feature_name_left = feature_name_left.replace(' ', '')
            feature_name_right = feature_name_right.replace(' ', '')
            feature_name_left = Xnames2Featurenames[feature_name_left]
            feature_name_right = Xnames2Featurenames[feature_name_right]
            feat_for_vis[f'{feature_name_left}\nvs.\n{feature_name_right}'] = {'importance': data_dict[k]['importance']}
        else:
            feat_for_vis[Xnames2Featurenames[k]] = {'importance': data_dict[k]['importance']}

    feature_importance_visualize(feat_for_vis, save_png=True, folder='.', name='gaminet_feat_imp')

def LR(X, y, dataset_name, model_name='LR'):
    m = Ridge()
    # if task == 'regression':
    # else:
    #     m = LogisticRegression()
    m.fit(X, y)
    import seaborn as sns
    #plot = sns.distplot(m.coef_)
    word_to_coef = dict(zip(m.feature_names_in_, m.coef_.squeeze()))
    dict(sorted(word_to_coef.items(), key=lambda item: item[1]))
    word_to_coef_df = pd.DataFrame.from_dict(word_to_coef, orient='index')
    print(word_to_coef_df)

    for i, feature_name in enumerate(X.columns):
        inp = torch.linspace(X[feature_name].min(), X[feature_name].max(), 1000)
        outp = word_to_coef[feature_name] *  inp #+ m.intercept_
        # outp = nam_model.feature_nns[i](inp).detach().numpy().squeeze()
        # if len(X[feature_name].unique()) == 2:
        #     make_one_hot_plot(outp[0], outp[-1], feature_name, model_name, dataset_name)
        # else:
        make_plot(inp, outp, outp, outp, feature_name, model_name, dataset_name)

# EBM_show(X, y) # for EBM_Show copy paste this script into a jupyter notebook and only run the EBM_Show dashboard
EBM(X, y, dataset_name)
GAM(X, y, dataset_name)
Gaminet(X, y, dataset_name)
LR(X, y, dataset_name)
# X.to_csv(f'export/X_full_{dataset_name}.csv')
# pd.DataFrame(y).to_csv(f'export/y_full_{dataset_name}.csv')
