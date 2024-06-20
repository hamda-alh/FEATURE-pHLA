import os
import pickle
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import utils
from sklearn import model_selection
from sklearn.model_selection import KFold
import scipy
#import xgboost as xgb
import matplotlib.pyplot as plt
import peptides

######

def extract_features(sequence):
    peptide = peptides.Peptide(sequence)
    features = peptide.descriptors()  # Get basic descriptors
    features.update({'boman': peptide.boman()})
    features.update({'hydrophobicity': peptide.hydrophobicity()})
    features.update({'charge':peptide.charge()})
    features.update({'molecular_weight':peptide.molecular_weight()})
    features.update({'aliphatic_index':peptide.aliphatic_index()})
    features.update({'instability_index':peptide.instability_index()})
    features.update({'structural_class':peptide.structural_class()})
    return features

#####

## following the conditions from MATHLA to categorize BA

def categorize(value):
    if value < 100:
        return 'positive-high'
    elif value < 500:
        return 'positive'
    elif value < 1000:
        return 'positive-intermediate'
    elif value < 5000:
        return 'positive-low'
    else:
       return 'negative'
    


    

#####

## Convert the labels to positive or negative
def label_category(row):
    if row['category'] in ['positive', 'positive-high', 'positive-intermediate']:
        return 'positive'
    else:
        return 'negative'

##### 

## Following MATHLA proposed formula to normalize the original nanomolar affinity between 0 and 1.

def normalize_measurement_value(measurement_value):
    """
    Normalize a series of measurement values. 
    The normalized values are ensured to be between 0 and 1.

    a_{normal} = 1 - \log_{50000}(a_{nM})
    """
    
    normalized_values = measurement_value.copy()

    
    positive_values = measurement_value > 0
    normalized_values[positive_values] = 1 - np.log(measurement_value[positive_values]) / np.log(50000)

    
    normalized_values[measurement_value == 0.0] = 0

    # Ensure the normalized values are within bounds [0, 1]
    normalized_values = normalized_values.clip(lower=0, upper=1)

    return normalized_values

### Get classification metrics
def calculate_classification_metrics(labels, predictions):
    return round(metrics.accuracy_score(labels,predictions),3),\
           round(metrics.f1_score(labels,predictions),3),\
           round(metrics.roc_auc_score(labels,predictions),3),\
           round(metrics.average_precision_score(labels,predictions),3)




### Build supervised learning model
def supervised_learning_steps(method, scoring, data_type, task, model, params, X_train, y_train, n_iter, n_splits = 5):

    gs = grid_search_cv(model, params, X_train, y_train, scoring=scoring, n_iter = n_iter, n_splits = n_splits)

    y_pred = gs.predict(X_train)
    y_pred[y_pred < 0] = 0

    if task:
        results=calculate_classification_metrics(y_train, y_pred)
        print("Acc: %.3f, F1: %.3f, AUC: %.3f, AUPR: %.3f" % (results[0], results[1], results[2], results[3]))
    else:
        results=calculate_regression_metrics(y_train,y_pred)
        print("MAE: %.3f, MSE: %.3f, R2: %.3f, Pearson R: %.3f, Spearman R: %.3f" % (results[0], results[1], results[2], results[3], results[4]))

    print('Parameters')
    print('----------')
    for p,v in gs.best_estimator_.get_params().items():
        print(p, ":", v)
    print('-' * 80)

    if task:
        save_model(gs, "%s_models/%s_%s_classifier_gs.pk" % (method,method,data_type))
        save_model(gs.best_estimator_, "%s_models/%s_%s_classifier_best_estimator.pk" %(method,method,data_type))
    else:
        save_model(gs, "%s_models/%s_%s_regressor_gs.pk" % (method,method,data_type))
        save_model(gs.best_estimator_, "%s_models/%s_%s_regressor_best_estimator.pk" %(method,method,data_type))
    return(gs)


def grid_search_cv(model, parameters, X_train, y_train, n_splits=5, n_iter=1000, n_jobs=42, scoring="r2", stratified=False):
    """
        Tries all possible values of parameters and returns the best regressor/classifier.
        Cross Validation done is stratified.
        See scoring options at https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    """

    # Stratified n_splits Folds. Shuffle is not needed as X and Y were already shuffled before.
    if stratified:
        cv = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=False, random_state=42)
    else:
        cv = n_splits

    rev_model = model_selection.RandomizedSearchCV(estimator=model, param_distributions=parameters, cv=cv, scoring=scoring, n_iter=n_iter, n_jobs=42, random_state=0, verbose=-1)
    if (model=="xgb"):
        xgbtrain = xgb.DMatrix(X_train, Y_train)
        output = rev_model.fit(xgbtrain)
        rm(xgbtrain)
        gc()
        return output
    else:
        return rev_model.fit(X_train, y_train)


def save_model(model, filename):

    outpath = os.path.join("../Models/", filename)

    with open(outpath, "wb") as f:
        pickle.dump(model, f)

    print("Saved model to file: %s" % (outpath))

####


def load_model(filename):

    fpath = os.path.join("../Models/", filename)

    with open(fpath, "rb") as f:
        model = pickle.load(f)

    print("Load model to file: %s" % (fpath))
    return model

####
def binary_label_category(row):
    if row['label'] in ['positive', 'positive-high', 'positive-intermediate']:
        return 1
    else:
        return 0