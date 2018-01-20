#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 22:20:36 2018

@author: Zhenyu
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split, KFold, cross_val_score

# Load normalized data
train_data=pickle.load(open('/Users/Zhenyu/Desktop/CentraleSupelec/Machine Learning/Projet/train_data.sav', 'rb'))
test_data=pickle.load(open('/Users/Zhenyu/Desktop/CentraleSupelec/Machine Learning/Projet/test_data.sav', 'rb'))

# Model
model = GradientBoostingRegressor()
X = train_data.drop(["power_increase"], axis=1)
y = train_data["power_increase"]

import matplotlib.pyplot as plt

# Learning curve
def plot_learning_curves(estimator, X, y, scoring="accuracy", cv=None, n_jobs=1, train_sizes=np.linspace(0.1,1.0,5)):
    """ Generate a plot showing training and test learning curves
        source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

    Parameters
    ----------
    estimator: object type
        the estimator that will be used to implement "fit" and "predict"

    X: array, shape(n_samples, m_features)
        Training vector

    y: array, shape(n_samples)
        Target relative to X

     scoring:string
        The scoring method   

    cv: int
        Cross-validation splitting strategy

    n_jobs: int
        Number of jobs to run in parallel

    train_sizes: array, shape(n_ticks)
        Number of training examples that will be used to generate
        the learning curve
    """

    plt.figure()
    plt.title("Learning Curves\n")
    plt.xlabel("Training examples")
    plt.ylabel("Score ({})".format(scoring))
    plt.legend(loc="best")
    plt.grid()

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")

    plt.plot(train_sizes, train_scores_mean, "o-", color="r",
             label="Training score")

    plt.plot(train_sizes, test_scores_mean, "o-", color="g", 
            label="Cross-validation score")

    plt.show()
    
# Fit
model.fit(X, y)

# Save model data
pickle.dump(model, open('/Users/Zhenyu/Desktop/CentraleSupelec/Machine Learning/Projet/model_boosting.sav', 'wb'))

# Predict
y_pred = model.predict(test_data)
submission = pd.DataFrame()
submission["power_increase"] = y_pred
submission.to_csv("/Users/Zhenyu/Desktop/CentraleSupelec/Machine Learning/Projet/test_solution_sample.csv", index=True)