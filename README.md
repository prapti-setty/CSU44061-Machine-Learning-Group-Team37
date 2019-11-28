# CSU44061-Machine-Learning-Group-Project

Group Members: Prapti Setty, Yash Pandey, Taranvir Singh

## Table of Contents
1. [About](#About)
1. [Method](#Method)
1. [Testing](#Testing)
1. [Other things tried](#Other)

## About
Machine Learning group competition as part of the 2019/20 machine learning module at Trinity College Dublin.
Predict the income of a person based on a number of features - follows on from individual competition completed as previous assignment for same module

## Method
Algorithm Used: LGBMRegressor & XGBoost

Data Exploration: Seaborn & Matplot

Preprocessing:

Nans - imputing

Encoding - Label Encoding

Other - convert non numeric data set to float,  get mean rank, log Income

Fitting model: 
params = {
          'max_depth': 20,
          'learning_rate': 0.01,
          "boosting": "gbdt",
          "bagging_seed": 11,
          "metric": 'mae',
          "verbosity": -1
         }

## Testing
We use simple MAE tester locally on patial dataset left out of training model to get estimated score. If we are please, then we train with entire dataset to predict for real test dataset and we submit.

## Other
