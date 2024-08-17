import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import warnings
from sklearn.decomposition import FactorAnalysis
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
import numpy
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
from imblearn.pipeline import Pipeline, make_pipeline
from skrebate import ReliefF
from sklearn.feature_selection import SelectFromModel
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import VotingClassifier
from skrebate import SURF
from skrebate import SURFstar
from skrebate import MultiSURF
from skrebate import MultiSURFstar
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from scipy.stats import uniform, randint
from scipy.stats import randint as sp_randint
import warnings
warnings.filterwarnings("ignore")
import sklearn.neighbors._base
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score
from scipy.stats import norm
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, MinMaxScaler,LabelEncoder
import time

start_time = time.time()

models = [RandomForestClassifier(random_state = 693), DecisionTreeClassifier(random_state = 693), XGBClassifier(random_state = 693), 
HistGradientBoostingClassifier(random_state = 693), GradientBoostingClassifier(random_state = 693), 
          ExtraTreesClassifier(random_state = 693), AdaBoostClassifier(random_state = 693), SVC(random_state = 693), LogisticRegression(random_state = 693),
         LinearDiscriminantAnalysis(), KNeighborsClassifier(), GaussianNB()]

X_validation = pd.read_csv('/home/vad3/Projet 3/Radiomics/data/testX_t0.csv')
X_validation.set_index(X_validation.columns[0], inplace=True)
Y_validation = pd.read_csv('/home/vad3/Projet 3/Radiomics/data/testY_t0.csv')
Y_validation.set_index(Y_validation.columns[0], inplace=True)
X_train = pd.read_csv('/home/vad3/Projet 3/Radiomics/data/trainX_t0.csv')
X_train.set_index(X_train.columns[0], inplace=True)
Y_train = pd.read_csv('/home/vad3/Projet 3/Radiomics/data/trainY_t0.csv')
Y_train.set_index(Y_train.columns[0], inplace=True)
## Same scale: MinMaxScaler(), StandardScaler()
def scale(df, method):
    df1 = df
    mms = method
    df_norm = mms.fit_transform(df)
    df_norm = pd.DataFrame( df_norm, columns=df1.columns, index=df1.index)
    return df_norm
X_train = scale(X_train, StandardScaler())
X_validation = scale(X_validation, StandardScaler())

X_train_init = X_train.copy()
X_validation_init = X_validation.copy()

cols = ['wavelet-LLH_firstorder_InterquartileRange',
       'wavelet-LHL_firstorder_Mean', 'wavelet-LHH_glszm_SmallAreaEmphasis',
       'wavelet-LHH_glszm_SmallAreaLowGrayLevelEmphasis',
       'wavelet-HLL_glcm_Correlation', 'wavelet-HLH_firstorder_Skewness',
       'wavelet-HHL_glszm_ZoneVariance',
       'wavelet-LLL_gldm_DependenceNonUniformity',
       'squareroot_gldm_DependenceNonUniformity',
       'gradient_firstorder_90Percentile', 'gradient_glcm_DifferenceVariance']

X_train = X_train[cols]
X_validation = X_validation[cols]
X_train_init = X_train.copy()
X_validation_init = X_validation.copy()

def AUC_with95CI_calculator(y, y_pred):
    fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=1)
    auroc = auc(fpr, tpr)
    specificity_sensitivity_sum = tpr + (1 - fpr)
    ind_max = np.argmax(specificity_sensitivity_sum)
    n_bootstrap = 1000
    auc_values = []
    for _ in range(n_bootstrap):
        y_true_bootstrap, y_scores_bootstrap = resample(y, y_pred, replace=True)
        try:
            AUC_temp = roc_auc_score(y_true_bootstrap, y_scores_bootstrap)
        except:
            continue
        auc_values.append(AUC_temp)
    lower_auroc = np.percentile(auc_values, 2.5)
    upper_auroc = np.percentile(auc_values, 97.5)
    return auroc, threshold[ind_max], lower_auroc, upper_auroc

print('----------------------------------------RELIEF--------------------------------------------------')
np.random.seed(42)

param1 = {
        "model__max_depth": [2, 3, 5, 10],
        "model__n_estimators": [50, 60, 90],
        "model__min_samples_leaf": [3, 5, 8],  
        "model__min_samples_split": [3, 5, 8],
        "model__max_features": ['sqrt', 'log2', 0.5, 0.8],
        "model__bootstrap": [True, False],
        "model__random_state": [693],
        "fs__n_features_to_select": list(range(3, 12))
    }
param2 = {
        "model__max_depth": [2, 3, 5, 10],
        "model__min_samples_leaf": [3, 5, 8],     
        "model__min_samples_split": [3, 5, 8],  
        "model__max_features": ['sqrt', 'log2', 0.5, 0.8],
        "model__criterion": ['gini', 'entropy'],
        "model__random_state": [693],
        "fs__n_features_to_select": list(range(3, 12))
    }
param3 = {
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__n_estimators": [50, 60, 90],
    "model__max_depth": [2, 3, 5, 10],
    "model__min_child_weight": [1, 2, 3, 4],
    "model__subsample": [0.5, 0.6, 0.7, 0.8],
    "model__colsample_bytree": [0.5, 0.6, 0.7, 0.8],
    "model__gamma": [2.5, 3, 4, 4.5],
    "model__random_state": [693],
    "fs__n_features_to_select": list(range(3, 12))
    }
param4 = {
        "model__max_iter": [50, 60, 90],
        "model__min_samples_leaf": [3, 5, 8],
        "model__max_bins":[10, 15, 20, 30, 35, 50],
        "model__learning_rate": [0.01, 0.05, 0.1], 
        "model__max_depth": [2, 3, 5, 10],
        "model__random_state": [693],
        "fs__n_features_to_select": list(range(3, 12))
    }
param5 = {
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__max_depth": [2, 3, 5, 10],
        "model__n_estimators": [50, 60, 90],
        "model__min_samples_leaf": [3, 5, 8],   
        "model__min_samples_split": [3, 5, 8],
        "model__max_features": ['sqrt', 'log2', 0.5, 0.8],
        "model__random_state": [693],
        "fs__n_features_to_select": list(range(3, 12))
    }
param6 = {
        "model__max_depth": [2, 3, 5, 10],
        "model__n_estimators": [50, 60, 90],
        "model__min_samples_leaf": [3, 5, 8], 
        "model__min_samples_split": [3, 5, 8], 
        "model__max_features": ['sqrt', 'log2', 0.5, 0.8],
        "model__bootstrap": [True, False],
        "model__random_state": [693],
        "fs__n_features_to_select": list(range(3, 12))
    }
param7 = {
        
        "model__n_estimators": [50, 60, 90],
        "model__learning_rate": [0.01, 0.1, 0.5, 1],
        "model__random_state": [693],
        "fs__n_features_to_select": list(range(3, 12))
        
    }
param8 = {
        
        "model__C": np.arange(0.1, 1., 0.1),
        "model__gamma": np.arange(0.001, 1., 0.1),
        "model__kernel": ['rbf','poly','linear'],
        "model__degree":[2,3,4,5],
        "model__random_state": [693],
        "model__probability":[True],
        "fs__n_features_to_select": list(range(3, 12))
        
    }
param9 = {
        
        "model__C": np.arange(0.1, 1., 0.1),
        "model__penalty": ['l1', 'l2'],
        "model__solver": ['liblinear', 'saga'],
        "model__max_iter" : [100, 1000, 2500, 5000],
        "model__random_state": [693],
        "fs__n_features_to_select": list(range(3, 12))
        
    }
param10 = {
        
        "model__solver": ['lsqr', 'eigen'],
        "model__shrinkage": [None, 'auto', 0.1, 0.5, 0.9],
        "fs__n_features_to_select": list(range(3, 12))
    }
param11 = {
        
        "model__n_neighbors": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "model__weights": [ 'distance'],
        "model__metric": ['euclidean'],
        "fs__n_features_to_select": list(range(3, 12))
        
    }
param12 = {
        
      # 'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
       "model__var_smoothing": np.logspace(0,-9, num=50),
        "fs__n_features_to_select": list(range(3, 12))    
    }


param_model = [param1, param2, param3, param4, param5, param6, param7, param8, param9, param10, param11, param12]
X_train = X_train_init.copy()
X_validation = X_validation_init.copy()
best_scores = []
validation_scores = []
n_splits = 5  # Nombre de plis pour la validation croisée
for model, params in zip(models, param_model):
    
    pipe = Pipeline([
            ('fs', ReliefF(n_neighbors=5)),
            ('smote', SVMSMOTE(sampling_strategy='minority', random_state=42)),
            ('model', model)
        ])
        # Configurer RandomizedSearchCV
    clf = GridSearchCV(pipe, params, cv = StratifiedKFold(n_splits=5), scoring='roc_auc',verbose=1)
    #clf = RandomizedSearchCV(pipe, params, random_state = 693, scoring=make_scorer(roc_auc_score), n_iter = 500, cv=StratifiedKFold(n_splits=5))
    clf.fit(X_train.values, Y_train['Label'].values)
    best_index = clf.best_index_
    cv_results = clf.cv_results_
    AUC_CV = cv_results['mean_test_score'][best_index]
    best_params = clf.best_params_
        # Ajouter le meilleur score et le nom du modèle à la liste best_scores
    best_scores.append({'Model': type(model).__name__, 'Best Score': AUC_CV})
    #print(f"Meilleurs paramètres: {best_params}, Meilleur score AUC: {best_score}")
    print(f"Meilleurs paramètres pour {type(model).__name__}: {best_params}, \n AUC in CV: {AUC_CV:.4f}")
   # print('Number of features', feature)

    # Préparer les paramètres du modèle
    model_params = {k.split("__")[1]: v for k, v in best_params.items() if k.startswith('model__')}

    # Création du pipeline final
    final_model = type(model)(**model_params)
    final_pipe = Pipeline([
        ('fs', ReliefF(n_neighbors=5, n_features_to_select=best_params['fs__n_features_to_select'])),
        ('smote', SVMSMOTE(sampling_strategy='minority', random_state=42)),
        ('model', final_model)])

    final_pipe.fit(X_train.values, np.ravel(Y_train['Label'].values))

    y_pred_validation = final_pipe.predict_proba(X_validation.values)[:, 1]
    final_validation_score = roc_auc_score(Y_validation['Label'].values, y_pred_validation)
    # Ajouter le score de validation et le nom du modèle à la liste validation_scores
    validation_scores.append({'Model': type(model).__name__, 'Validation Score': final_validation_score})
    print(f"Score AUC sur l'ensemble de validation pour {type(model).__name__} : {final_validation_score:.4f}")
    x1, x2, x3, x4 = AUC_with95CI_calculator(Y_validation, y_pred_validation)
    print('lower', x3)
    print('upper', x4)
    #print(f"Score AUC sur l'ensemble de validation: {final_validation_score}")
df_best_scoresRl = pd.DataFrame(best_scores)
df_best_scoresRl.set_index(df_best_scoresRl.columns[0], inplace=True)
df_best_scoresRl.rename(columns={'Best Score': 'RL'}, inplace=True)
df_validation_scoresRl = pd.DataFrame(validation_scores)
df_validation_scoresRl.set_index(df_validation_scoresRl.columns[0], inplace=True)
df_validation_scoresRl.rename(columns={'Validation Score': 'RL'}, inplace=True)

print(df_best_scoresRl)
print(df_validation_scoresRl)
print('------------------------------------------------------------------------------------------')


print('----------------------------------------SURF--------------------------------------------------')
np.random.seed(42)
X_train = X_train_init.copy()
X_validation = X_validation_init.copy()
best_scores = []
validation_scores = []
n_splits = 5  # Nombre de plis pour la validation croisée
for model, params in zip(models, param_model):
        # Créer un pipeline avec ReliefF, SVMSMOTE et le modèle
    pipe = Pipeline([
            ('fs', SURF()),
            ('smote', SVMSMOTE(sampling_strategy='minority', random_state=42)),
            ('model', model)
        ])
    clf = GridSearchCV(pipe, params, cv = StratifiedKFold(n_splits=5), scoring='roc_auc',verbose=1)
    #clf = RandomizedSearchCV(pipe, params, random_state = 693, scoring=make_scorer(roc_auc_score), n_iter = 500, cv=StratifiedKFold(n_splits=5))
    clf.fit(X_train.values, Y_train['Label'].values)
    best_index = clf.best_index_
    cv_results = clf.cv_results_
    AUC_CV = cv_results['mean_test_score'][best_index]
    best_params = clf.best_params_
        # Ajouter le meilleur score et le nom du modèle à la liste best_scores
    best_scores.append({'Model': type(model).__name__, 'Best Score': AUC_CV})
    #print(f"Meilleurs paramètres: {best_params}, Meilleur score AUC: {best_score}")
    print(f"Meilleurs paramètres pour {type(model).__name__}: {best_params}, \n AUC in CV: {AUC_CV:.4f}")

    # Préparer les paramètres du modèle
    model_params = {k.split("__")[1]: v for k, v in best_params.items() if k.startswith('model__')}

    # Création du pipeline final
    final_model = type(model)(**model_params)
    final_pipe = Pipeline([
        ('fs', SURF(n_features_to_select=best_params['fs__n_features_to_select'])),
        ('smote', SVMSMOTE(sampling_strategy='minority', random_state=42)),
        ('model', final_model)])

    final_pipe.fit(X_train.values, np.ravel(Y_train['Label'].values))

    y_pred_validation = final_pipe.predict_proba(X_validation.values)[:, 1]
    final_validation_score = roc_auc_score(Y_validation['Label'].values, y_pred_validation)
    # Ajouter le score de validation et le nom du modèle à la liste validation_scores
    validation_scores.append({'Model': type(model).__name__, 'Validation Score': final_validation_score})
    print(f"Score AUC sur l'ensemble de validation pour {type(model).__name__} : {final_validation_score:.4f}")
    x1, x2, x3, x4 = AUC_with95CI_calculator(Y_validation, y_pred_validation)
    print('lower', x3)
    print('upper', x4)
    #print(f"Score AUC sur l'ensemble de validation: {final_validation_score}")
df_best_scoressurf = pd.DataFrame(best_scores)
df_best_scoressurf.set_index(df_best_scoressurf.columns[0], inplace=True)
df_best_scoressurf.rename(columns={'Best Score': 'SURF'}, inplace=True)
df_validation_scoressurf = pd.DataFrame(validation_scores)
df_validation_scoressurf.set_index(df_validation_scoressurf.columns[0], inplace=True)
df_validation_scoressurf.rename(columns={'Validation Score': 'SURF'}, inplace=True)
print(df_best_scoressurf)
print(df_validation_scoressurf)
print('------------------------------------------------------------------------------------------')


print('----------------------------------------SURFSTAR--------------------------------------------------')
np.random.seed(42)
X_train = X_train_init.copy()
X_validation = X_validation_init.copy()
best_scores = []
validation_scores = []
n_splits = 5  # Nombre de plis pour la validation croisée
for model, params in zip(models, param_model):
        # Créer un pipeline avec ReliefF, SVMSMOTE et le modèle
    pipe = Pipeline([
            ('fs', SURFstar()),
            ('smote', SVMSMOTE(sampling_strategy='minority', random_state=42)),
            ('model', model)])

    clf = GridSearchCV(pipe, params, cv = StratifiedKFold(n_splits=5), scoring='roc_auc',verbose=1)
    #clf = RandomizedSearchCV(pipe, params, random_state = 693, scoring=make_scorer(roc_auc_score), n_iter = 500, cv=StratifiedKFold(n_splits=5))
    clf.fit(X_train.values, Y_train['Label'].values)
    best_index = clf.best_index_
    cv_results = clf.cv_results_
    AUC_CV = cv_results['mean_test_score'][best_index]
    best_params = clf.best_params_
        # Ajouter le meilleur score et le nom du modèle à la liste best_scores
    best_scores.append({'Model': type(model).__name__, 'Best Score': AUC_CV})
    #print(f"Meilleurs paramètres: {best_params}, Meilleur score AUC: {best_score}")
    print(f"Meilleurs paramètres pour {type(model).__name__}: {best_params}, \n AUC in CV: {AUC_CV:.4f}")

    # Préparer les paramètres du modèle
    model_params = {k.split("__")[1]: v for k, v in best_params.items() if k.startswith('model__')}

    # Création du pipeline final
    final_model = type(model)(**model_params)
    final_pipe = Pipeline([
        ('fs', SURFstar( n_features_to_select=best_params['fs__n_features_to_select'])),
        ('smote', SVMSMOTE(sampling_strategy='minority', random_state=42)),
        ('model', final_model)])

    final_pipe.fit(X_train.values, np.ravel(Y_train['Label'].values))

    y_pred_validation = final_pipe.predict_proba(X_validation.values)[:, 1]
    final_validation_score = roc_auc_score(Y_validation['Label'].values, y_pred_validation)
    # Ajouter le score de validation et le nom du modèle à la liste validation_scores
    validation_scores.append({'Model': type(model).__name__, 'Validation Score': final_validation_score})
    print(f"Score AUC sur l'ensemble de validation pour {type(model).__name__} : {final_validation_score:.4f}")
    x1, x2, x3, x4 = AUC_with95CI_calculator(Y_validation, y_pred_validation)
    print('lower', x3)
    print('upper', x4)
    #print(f"Score AUC sur l'ensemble de validation: {final_validation_score}")
df_best_scoressurfs = pd.DataFrame(best_scores)
df_best_scoressurfs.set_index(df_best_scoressurfs.columns[0], inplace=True)
df_best_scoressurfs.rename(columns={'Best Score': 'SURFS'}, inplace=True)
df_validation_scoressurfs = pd.DataFrame(validation_scores)
df_validation_scoressurfs.set_index(df_validation_scoressurfs.columns[0], inplace=True)
df_validation_scoressurfs.rename(columns={'Validation Score': 'SURFS'}, inplace=True)
print(df_best_scoressurfs)
print(df_validation_scoressurfs)
print('------------------------------------------------------------------------------------------')
print('----------------------------------------MULTISURF--------------------------------------------------')
np.random.seed(42)
X_train = X_train_init.copy()
X_validation = X_validation_init.copy()
best_scores = []
validation_scores = []
n_splits = 5  # Nombre de plis pour la validation croisée
for model, params in zip(models, param_model):
        # Créer un pipeline avec ReliefF, SVMSMOTE et le modèle
    pipe = Pipeline([
            ('fs', MultiSURF()),
            ('smote', SVMSMOTE(sampling_strategy='minority', random_state=42)),
            ('model', model)])
    
    clf = GridSearchCV(pipe, params, cv = StratifiedKFold(n_splits=5), scoring='roc_auc',verbose=1)
    #clf = RandomizedSearchCV(pipe, params, random_state = 693, scoring=make_scorer(roc_auc_score), n_iter = 500, cv=StratifiedKFold(n_splits=5))
    clf.fit(X_train.values, Y_train['Label'].values)
    best_index = clf.best_index_
    cv_results = clf.cv_results_
    AUC_CV = cv_results['mean_test_score'][best_index]
    best_params = clf.best_params_
        # Ajouter le meilleur score et le nom du modèle à la liste best_scores
    best_scores.append({'Model': type(model).__name__, 'Best Score': AUC_CV})
    #print(f"Meilleurs paramètres: {best_params}, Meilleur score AUC: {best_score}")
    print(f"Meilleurs paramètres pour {type(model).__name__}: {best_params}, \n AUC in CV: {AUC_CV:.4f}")
   

    # Préparer les paramètres du modèle
    model_params = {k.split("__")[1]: v for k, v in best_params.items() if k.startswith('model__')}

    # Création du pipeline final
    final_model = type(model)(**model_params)
    final_pipe = Pipeline([
        ('fs', MultiSURF( n_features_to_select=best_params['fs__n_features_to_select'])),
        ('smote', SVMSMOTE(sampling_strategy='minority', random_state=42)),
        ('model', final_model)])

    final_pipe.fit(X_train.values, np.ravel(Y_train['Label'].values))

    y_pred_validation = final_pipe.predict_proba(X_validation.values)[:, 1]
    final_validation_score = roc_auc_score(Y_validation['Label'].values, y_pred_validation)
    # Ajouter le score de validation et le nom du modèle à la liste validation_scores
    validation_scores.append({'Model': type(model).__name__, 'Validation Score': final_validation_score})
    print(f"Score AUC sur l'ensemble de validation pour {type(model).__name__} : {final_validation_score:.4f}")
    x1, x2, x3, x4 = AUC_with95CI_calculator(Y_validation, y_pred_validation)
    print('lower', x3)
    print('upper', x4)
    #print(f"Score AUC sur l'ensemble de validation: {final_validation_score}")
df_best_scoresmsurf = pd.DataFrame(best_scores)
df_best_scoresmsurf.set_index(df_best_scoresmsurf.columns[0], inplace=True)
df_best_scoresmsurf.rename(columns={'Best Score': 'MSURF'}, inplace=True)
df_validation_scoresmsurf = pd.DataFrame(validation_scores)
df_validation_scoresmsurf.set_index(df_validation_scoresmsurf.columns[0], inplace=True)
df_validation_scoresmsurf.rename(columns={'Validation Score': 'MSURF'}, inplace=True)
print(df_best_scoresmsurf)
print(df_validation_scoresmsurf)
print('------------------------------------------------------------------------------------------')

print('----------------------------------------MULTISURFSTAR--------------------------------------------------')
np.random.seed(42)
X_train = X_train_init.copy()
X_validation = X_validation_init.copy()
best_scores = []
validation_scores = []
n_splits = 5  # Nombre de plis pour la validation croisée
for model, params in zip(models, param_model):
        
    pipe = Pipeline([
            ('fs', MultiSURFstar()),
            ('smote', SVMSMOTE(sampling_strategy='minority', random_state=42)),
            ('model', model)
        ])
    clf = GridSearchCV(pipe, params, cv = StratifiedKFold(n_splits=5), scoring='roc_auc',verbose=1)
    #clf = RandomizedSearchCV(pipe, params, random_state = 693, scoring=make_scorer(roc_auc_score), n_iter = 500, cv=StratifiedKFold(n_splits=5))
    clf.fit(X_train.values, Y_train['Label'].values)
    best_index = clf.best_index_
    cv_results = clf.cv_results_
    AUC_CV = cv_results['mean_test_score'][best_index]
    best_params = clf.best_params_
        # Ajouter le meilleur score et le nom du modèle à la liste best_scores
    best_scores.append({'Model': type(model).__name__, 'Best Score': AUC_CV})
    #print(f"Meilleurs paramètres: {best_params}, Meilleur score AUC: {best_score}")
    print(f"Meilleurs paramètres pour {type(model).__name__}: {best_params}, \n AUC in CV: {AUC_CV:.4f}")
    # Préparer les paramètres du modèle
    model_params = {k.split("__")[1]: v for k, v in best_params.items() if k.startswith('model__')}

    # Création du pipeline final
    final_model = type(model)(**model_params)
    final_pipe = Pipeline([
        ('fs', MultiSURFstar(n_features_to_select=best_params['fs__n_features_to_select'])),
        ('smote', SVMSMOTE(sampling_strategy='minority', random_state=42)),
        ('model', final_model)])

    final_pipe.fit(X_train.values, np.ravel(Y_train['Label'].values))

    y_pred_validation = final_pipe.predict_proba(X_validation.values)[:, 1]
    final_validation_score = roc_auc_score(Y_validation['Label'].values, y_pred_validation)
    # Ajouter le score de validation et le nom du modèle à la liste validation_scores
    validation_scores.append({'Model': type(model).__name__, 'Validation Score': final_validation_score})
    print(f"Score AUC sur l'ensemble de validation pour {type(model).__name__} : {final_validation_score:.4f}")
    x1, x2, x3, x4 = AUC_with95CI_calculator(Y_validation, y_pred_validation)
    print('lower', x3)
    print('upper', x4)
    #print(f"Score AUC sur l'ensemble de validation: {final_validation_score}")
df_best_scoresmsurfs = pd.DataFrame(best_scores)
df_best_scoresmsurfs.set_index(df_best_scoresmsurfs.columns[0], inplace=True)
df_best_scoresmsurfs.rename(columns={'Best Score': 'MSURFS'}, inplace=True)
df_validation_scoresmsurfs = pd.DataFrame(validation_scores)
df_validation_scoresmsurfs.set_index(df_validation_scoresmsurfs.columns[0], inplace=True)
df_validation_scoresmsurfs.rename(columns={'Validation Score': 'MSURFS'}, inplace=True)
print(df_best_scoresmsurfs)
print(df_validation_scoresmsurfs)
print('------------------------------------------------------------------------------------------')

print('----------------------------------------SFS--------------------------------------------------')
np.random.seed(42)
X_train = X_train_init.copy()
X_validation = X_validation_init.copy()
best_scores = []
validation_scores = []
n_splits = 5  # Nombre de plis pour la validation croisée
lr = LogisticRegression(penalty='l1',solver='liblinear', max_iter=10000, class_weight='balanced', C = 0.1,  random_state = 693)
for model, params in zip(models, param_model):
    
    pipe = Pipeline([
        ('fs', SequentialFeatureSelector(lr)),
        ('smote', SVMSMOTE(sampling_strategy='minority', random_state=42)),
        ('model', model)
    ])
    #clf = RandomizedSearchCV(pipe, params, random_state = 693, scoring=make_scorer(roc_auc_score), n_iter = 500, cv=StratifiedKFold(n_splits=5))
    clf = GridSearchCV(pipe, params, cv = StratifiedKFold(n_splits=5), scoring='roc_auc',verbose=1)
    clf.fit(X_train.values, Y_train['Label'].values)
    best_index = clf.best_index_
    cv_results = clf.cv_results_
    AUC_CV = cv_results['mean_test_score'][best_index]
    best_params = clf.best_params_
        # Ajouter le meilleur score et le nom du modèle à la liste best_scores
    best_scores.append({'Model': type(model).__name__, 'Best Score': AUC_CV})
    #print(f"Meilleurs paramètres: {best_params}, Meilleur score AUC: {best_score}")
    print(f"Meilleurs paramètres pour {type(model).__name__}: {best_params}, \n AUC in CV: {AUC_CV:.4f}")
    # Préparer les paramètres du modèle
    model_params = {k.split("__")[1]: v for k, v in best_params.items() if k.startswith('model__')}

    # Création du pipeline final
    final_model = type(model)(**model_params)
    final_pipe = Pipeline([
        ('fs', SequentialFeatureSelector(lr, n_features_to_select=best_params['fs__n_features_to_select'])),
        ('smote', SVMSMOTE(sampling_strategy='minority', random_state=42)),
        ('model', final_model)])

    final_pipe.fit(X_train.values, np.ravel(Y_train['Label'].values))

    y_pred_validation = final_pipe.predict_proba(X_validation.values)[:, 1]
    final_validation_score = roc_auc_score(Y_validation['Label'].values, y_pred_validation)
    # Ajouter le score de validation et le nom du modèle à la liste validation_scores
    validation_scores.append({'Model': type(model).__name__, 'Validation Score': final_validation_score})
    print(f"Score AUC sur l'ensemble de validation pour {type(model).__name__} : {final_validation_score:.4f}")
    x1, x2, x3, x4 = AUC_with95CI_calculator(Y_validation, y_pred_validation)
    print('lower', x3)
    print('upper', x4)
    #print(f"Score AUC sur l'ensemble de validation: {final_validation_score}")
df_best_scoressfs = pd.DataFrame(best_scores)
df_best_scoressfs.set_index(df_best_scoressfs.columns[0], inplace=True)
df_best_scoressfs.rename(columns={'Best Score': 'SFS'}, inplace=True)
df_validation_scoressfs = pd.DataFrame(validation_scores)
df_validation_scoressfs.set_index(df_validation_scoressfs.columns[0], inplace=True)
df_validation_scoressfs.rename(columns={'Validation Score': 'SFS'}, inplace=True)
print(df_best_scoressfs)
print(df_validation_scoressfs)
print('------------------------------------------------------------------------------------------')



param1 = {
        "model__max_depth": [2, 3, 5, 10],
        "model__n_estimators": [50, 60, 90],
        "model__min_samples_leaf": [3, 5, 8],  
        "model__min_samples_split": [3, 5, 8],
        "model__max_features": ['sqrt', 'log2', 0.5, 0.8],
        "model__bootstrap": [True, False],
        "model__random_state": [693],
        "fs2__k": list(range(3, 12))
    }
param2 = {
        "model__max_depth": [2, 3, 5, 10],
        "model__min_samples_leaf": [3, 5, 8],     
        "model__min_samples_split": [3, 5, 8],  
        "model__max_features": ['sqrt', 'log2', 0.5, 0.8],
        "model__criterion": ['gini', 'entropy'],
        "model__random_state": [693],
        "fs2__k": list(range(3, 12))
    }
param3 = {
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__n_estimators": [50, 60, 90],
    "model__max_depth": [2, 3, 5, 10],
    "model__min_child_weight": [1, 2, 3, 4],
    "model__subsample": [0.5, 0.6, 0.7, 0.8],
    "model__colsample_bytree": [0.5, 0.6, 0.7, 0.8],
    "model__gamma": [2.5, 3, 4, 4.5],
    "model__random_state": [693],
    "fs2__k": list(range(3, 12))
    }
param4 = {
        "model__max_iter": [50, 60, 90],
        "model__min_samples_leaf": [3, 5, 8],
        "model__max_bins":[10, 15, 20, 30, 35, 50],
        "model__learning_rate": [0.01, 0.05, 0.1], 
        "model__max_depth": [2, 3, 5, 10],
        "model__random_state": [693],
        "fs2__k": list(range(3, 12))
    }
param5 = {
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__max_depth": [2, 3, 5, 10],
        "model__n_estimators": [50, 60, 90],
        "model__min_samples_leaf": [3, 5, 8],   
        "model__min_samples_split": [3, 5, 8],
        "model__max_features": ['sqrt', 'log2', 0.5, 0.8],
        "model__random_state": [693],
        "fs2__k": list(range(3, 12))
    }
param6 = {
        "model__max_depth": [2, 3, 5, 10],
        "model__n_estimators": [50, 60, 90],
        "model__min_samples_leaf": [3, 5, 8], 
        "model__min_samples_split": [3, 5, 8], 
        "model__max_features": ['sqrt', 'log2', 0.5, 0.8],
        "model__bootstrap": [True, False],
        "model__random_state": [693],
        "fs2__k": list(range(3, 12))
    }
param7 = {
        
        "model__n_estimators": [50, 60, 90],
        "model__learning_rate": [0.01, 0.1, 0.5, 1],
        "model__random_state": [693],
        "fs2__k": list(range(3, 12))
        
    }
param8 = {
        
        "model__C": np.arange(0.1, 1., 0.1),
        "model__gamma": np.arange(0.001, 1., 0.1),
        "model__kernel": ['rbf','poly','linear'],
        "model__degree":[2,3,4,5],
        "model__random_state": [693],
        "model__probability":[True],
        "fs2__k": list(range(3, 12))
        
    }
param9 = {
        
        "model__C": np.arange(0.1, 1., 0.1),
        "model__penalty": ['l1', 'l2'],
        "model__solver": ['liblinear', 'saga'],
        "model__max_iter" : [100, 1000, 2500, 5000],
        "model__random_state": [693],
        "fs2__k": list(range(3, 12))
        
    }
param10 = {
        
        "model__solver": ['lsqr', 'eigen'],
        "model__shrinkage": [None, 'auto', 0.1, 0.5, 0.9],
        "fs2__k": list(range(3, 12))
    }
param11 = {
        
        "model__n_neighbors": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "model__weights": [ 'distance'],
        "model__metric": ['euclidean'],
        "fs2__k": list(range(3, 12))
        
    }
param12 = {
        
      # 'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
       "model__var_smoothing": np.logspace(0,-9, num=50),
        "fs2__k": list(range(3, 12))
    }


param_model = [param1, param2, param3, param4, param5, param6, param7, param8, param9, param10, param11, param12]
print('----------------------------------------ANOVA--------------------------------------------------')
np.random.seed(42)
X_train = X_train_init.copy()
X_validation = X_validation_init.copy()
best_scores = []
validation_scores = []
n_splits = 5  # Nombre de plis pour la validation croisée
for model, params in zip(models, param_model):
    
    pipe = Pipeline([
            ('fs2', SelectKBest(f_classif)),
            ('smote', SVMSMOTE(sampling_strategy='minority', random_state=42)),
            ('model', model)
        ])
        # Configurer RandomizedSearchCV
    clf = GridSearchCV(pipe, params, cv = StratifiedKFold(n_splits=5), scoring='roc_auc',verbose=1)
    #clf = RandomizedSearchCV(pipe, params, random_state = 693, scoring=make_scorer(roc_auc_score), n_iter = 500, cv=StratifiedKFold(n_splits=5))
    clf.fit(X_train.values, Y_train['Label'].values)
    best_index = clf.best_index_
    cv_results = clf.cv_results_
    AUC_CV = cv_results['mean_test_score'][best_index]
    best_params = clf.best_params_
        # Ajouter le meilleur score et le nom du modèle à la liste best_scores
    best_scores.append({'Model': type(model).__name__, 'Best Score': AUC_CV})
    #print(f"Meilleurs paramètres: {best_params}, Meilleur score AUC: {best_score}")
    print(f"Meilleurs paramètres pour {type(model).__name__}: {best_params}, \n AUC in CV: {AUC_CV:.4f}")
    
    # Préparer les paramètres du modèle
    model_params = {k.split("__")[1]: v for k, v in best_params.items() if k.startswith('model__')}

    # Création du pipeline final
    final_model = type(model)(**model_params)
    final_pipe = Pipeline([
        ('fs2', SelectKBest(f_classif, k=best_params['fs2__k'])),
        ('smote', SVMSMOTE(sampling_strategy='minority', random_state=42)),
        ('model', final_model)])

    final_pipe.fit(X_train.values, np.ravel(Y_train['Label'].values))

    y_pred_validation = final_pipe.predict_proba(X_validation.values)[:, 1]
    final_validation_score = roc_auc_score(Y_validation['Label'].values, y_pred_validation)
    # Ajouter le score de validation et le nom du modèle à la liste validation_scores
    validation_scores.append({'Model': type(model).__name__, 'Validation Score': final_validation_score})
    print(f"Score AUC sur l'ensemble de validation pour {type(model).__name__} : {final_validation_score:.4f}")
    x1, x2, x3, x4 = AUC_with95CI_calculator(Y_validation, y_pred_validation)
    print('lower', x3)
    print('upper', x4)
    #print(f"Score AUC sur l'ensemble de validation: {final_validation_score}")
df_best_scoresAFT = pd.DataFrame(best_scores)
df_best_scoresAFT.set_index(df_best_scoresAFT.columns[0], inplace=True)
df_best_scoresAFT.rename(columns={'Best Score': 'AFT'}, inplace=True)
df_validation_scoresAFT = pd.DataFrame(validation_scores)
df_validation_scoresAFT.set_index(df_validation_scoresAFT.columns[0], inplace=True)
df_validation_scoresAFT.rename(columns={'Validation Score': 'AFT'}, inplace=True)
print(df_best_scoresAFT)
print(df_validation_scoresAFT)
print('------------------------------------------------------------------------------------------')

print('----------------------------------------MUTUAL INFORMATION--------------------------------------------------')
np.random.seed(42)
X_train = X_train_init.copy()
X_validation = X_validation_init.copy()
best_scores = []
validation_scores = []
n_splits = 5  # Nombre de plis pour la validation croisée
for model, params in zip(models, param_model):
    
        # Créer un pipeline avec ReliefF, SVMSMOTE et le modèle
    pipe = Pipeline([
            ('fs2', SelectKBest(mutual_info_classif)),
            ('smote', SVMSMOTE(sampling_strategy='minority', random_state=42)),
            ('model', model)])
    
    clf = GridSearchCV(pipe, params, cv = StratifiedKFold(n_splits=5), scoring='roc_auc',verbose=1)
    #clf = RandomizedSearchCV(pipe, params, random_state = 693, scoring=make_scorer(roc_auc_score), n_iter = 500, cv=StratifiedKFold(n_splits=5))
    clf.fit(X_train.values, Y_train['Label'].values)
    best_index = clf.best_index_
    cv_results = clf.cv_results_
    AUC_CV = cv_results['mean_test_score'][best_index]
    best_params = clf.best_params_
        # Ajouter le meilleur score et le nom du modèle à la liste best_scores
    best_scores.append({'Model': type(model).__name__, 'Best Score': AUC_CV})
    #print(f"Meilleurs paramètres: {best_params}, Meilleur score AUC: {best_score}")
    print(f"Meilleurs paramètres pour {type(model).__name__}: {best_params}, \n AUC in CV: {AUC_CV:.4f}")

    # Préparer les paramètres du modèle
    model_params = {k.split("__")[1]: v for k, v in best_params.items() if k.startswith('model__')}

    # Création du pipeline final
    final_model = type(model)(**model_params)
    final_pipe = Pipeline([
        ('fs2', SelectKBest(mutual_info_classif, k=best_params['fs2__k'])),
        ('smote', SVMSMOTE(sampling_strategy='minority', random_state=42)),
        ('model', final_model)])

    final_pipe.fit(X_train.values, np.ravel(Y_train['Label'].values))

    y_pred_validation = final_pipe.predict_proba(X_validation.values)[:, 1]
    final_validation_score = roc_auc_score(Y_validation['Label'].values, y_pred_validation)
    # Ajouter le score de validation et le nom du modèle à la liste validation_scores
    validation_scores.append({'Model': type(model).__name__, 'Validation Score': final_validation_score})
    print(f"Score AUC sur l'ensemble de validation pour {type(model).__name__} : {final_validation_score:.4f}")
    x1, x2, x3, x4 = AUC_with95CI_calculator(Y_validation, y_pred_validation)
    print('lower', x3)
    print('upper', x4)
    #print(f"Score AUC sur l'ensemble de validation: {final_validation_score}")
df_best_scoresmi = pd.DataFrame(best_scores)
df_best_scoresmi.set_index(df_best_scoresmi.columns[0], inplace=True)
df_best_scoresmi.rename(columns={'Best Score': 'MI'}, inplace=True)
df_validation_scoresmi = pd.DataFrame(validation_scores)
df_validation_scoresmi.set_index(df_validation_scoresmi.columns[0], inplace=True)
df_validation_scoresmi.rename(columns={'Validation Score': 'MI'}, inplace=True)
print(df_best_scoresmi)
print(df_validation_scoresmi)
print('------------------------------------------------------------------------------------------')


auc_cv = pd.concat([df_best_scoresRl, df_best_scoresAFT, df_best_scoressurf, df_best_scoresmi, df_best_scoresmsurf, df_best_scoressurfs, df_best_scoressfs, df_best_scoresmsurfs], axis=1)
auc_valid = pd.concat([df_validation_scoresRl, df_validation_scoresAFT, df_validation_scoressurf, df_validation_scoresmi, df_validation_scoresmsurf, df_validation_scoressurfs, df_validation_scoressfs, df_validation_scoresmsurfs], axis=1)
print(auc_cv)
print(auc_valid)

auc_cv.to_excel('/home/vad3/Projet 3/Radiomics/data/auc_cv.xlsx', index=False)
auc_valid.to_excel('/home/vad3/Projet 3/Radiomics/data/auc_valid.xlsx', index=False)
end_time = time.time()
execution_time = end_time - start_time
print(f"The execution time of the code is : {execution_time} secondes.")