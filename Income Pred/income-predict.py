
# Load the Pandas libraries with alias pd
import pandas as pd
# Load the different libraries needed from sklearn
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
# Load numpy libraries with alias np
import numpy as np
from scipy import stats
import math
from math import sqrt

def deal_with_nan_training(data):
    data['University Degree'] = data['University Degree'].fillna('No')
    # Drop any rows which have any nan's
    data = data.dropna(axis=0)
    return data

def avg_age(data):
    avg = (data['Age'].mean())
    return avg

def avg_size_of_city(data):
    avg = (data['Size of City'].mean())
    return avg

def avg_year_of_record(data):
    avg = (data['Year of Record'].mean())
    return avg

def avg_body_height(data):
    avg = (data['Body Height [cm]'].mean())
    return avg

def avg_crime(data):
    avg = (data['Crime Level in the City of Employement'].mean())
    return avg

def avg_work_exp(data):
    avg = (data['Work Experience in Current Job [years]'].mean())
    return avg
    

def deal_with_nan_test(test_data, training_data):
    test_data['Gender'] = test_data['Gender'].fillna('unknown')
    test_data['University Degree'] = test_data['University Degree'].fillna('No')
    test_data['Age'] = test_data['Age'].fillna(int(avg_age(training_data)))
    test_data['Size of City'] = test_data['Size of City'].fillna(int(avg_size_of_city(training_data)))
    test_data['Year of Record'] = test_data['Year of Record'].fillna(int(avg_year_of_record(training_data)))
    test_data['Body Height [cm]'] = test_data['Body Height [cm]'].fillna(int(avg_body_height(training_data)))
    test_data['Profession'] = test_data['Profession'].fillna('Unknown')
    test_data['Crime Level in the City of Employement'] = test_data['Crime Level in the City of Employement'].fillna(int(avg_crime(training_data)))
    #test_data['Work Experience in Current Job [years]'] = test_data['Work Experience in Current Job [years]'].fillna(5)
    test_data['Satisfation with employer'] = test_data['Satisfation with employer'].fillna("Average")
    #test_data['Yearly Income in addition to Salary (e.g. Rental Income)'] = test_data['Yearly Income in addition to Salary (e.g. Rental Income)'].fillna(0)
    return test_data

def transform_features(data):
    data['University Degree'] = data['University Degree'].map({'Bachelor':1, 'PhD':3, 'other':0.5, '0':0, 'Master':2, 'No':0})
    encode = pd.get_dummies(data, columns=[ "Gender","Country","Profession","Housing Situation","Satisfation with employer"],
    prefix=["enc_gen","enc_country","enc_pro","enc_housing","enc_sat"], drop_first=True)
    return encode
    
def main():
    # Read data from files
    training_data1 = pd.read_csv("tcd-ml-1920-group-income-train (1).csv")
    training_data2 = pd.read_csv("tcd-ml-1920-group-income-train (2).csv")
    training_data = pd.concat((training_data1,training_data2), axis=1)
    test_data = pd.read_csv("tcd-ml-1920-group-income-test.csv")
    
    training_data = training_data.drop(['Hair Color', 'Wears Glasses','Work Experience in Current Job [years]','Yearly Income in addition to Salary (e.g. Rental Income)'], axis=1)
    test_data = test_data.drop(['Hair Color', 'Wears Glasses','Work Experience in Current Job [years]','Yearly Income in addition to Salary (e.g. Rental Income)'], axis=1)

    # Clean Data
    # Remove any Nan's from training data
    non_nan_data = deal_with_nan_training(training_data)
    # Fill Nan's in test data with averages
    non_nan_test_data = deal_with_nan_test(test_data,non_nan_data)

    split_data_at = non_nan_data.shape[0]
    print(split_data_at)
    
    # Join training and test data
    appended = non_nan_data.append(non_nan_test_data)

    # Label encode the entire dataset of training + test
    # for column in appended.columns:
    #     if appended[column].dtype == type(object):
    #         le = preprocessing.LabelEncoder()
    #         appended[column] = le.fit_transform(appended[column])
    
    # One hot encode the entire dataset of training + test
    transformed_data = transform_features(appended)
    
    # Split data back into training and test seperately
    df_1 = pd.DataFrame()
    df_2 = pd.DataFrame()
    if appended.shape[0] > split_data_at: 
        df_1 = transformed_data[:split_data_at]
        df_2 = transformed_data[split_data_at:]

    # Create  model (Linear Regression)
    model = RandomForestRegressor(n_estimators=10)
    model2 = linear_model.Ridge(alpha = 0.001, normalize=True, fit_intercept=True)

    train = df_1
    test = df_2
    
    y_train = pd.DataFrame({'Income':train['Total Yearly Income [EUR]']})
    
    instance = test['Instance'].values.reshape(-1,1)

    #avg_income = (train['Income in EUR'].mean())

    train = train.drop(['Total Yearly Income [EUR]','Instance','#N/A', '0 EUR', '1', '116', '15', '156', '16650.13', '2000', '33', '44313', '743135', 'Average', 'Brown', 'Congo', 'Large Apartment', 'other', 'real estate manager'], axis=1)
    test = test.drop(['Total Yearly Income [EUR]','Instance','#N/A', '0 EUR', '1', '116', '15', '156', '16650.13', '2000', '33', '44313', '743135', 'Average', 'Brown', 'Congo', 'Large Apartment', 'other', 'real estate manager'], axis=1)

    X_train = train
    X_test = test

    A_train, A_test, b_train, b_test = train_test_split(X_train, y_train, test_size=0.1, random_state = 10)
    
    a = X_train.columns[X_train.isna().any()].tolist()
    print(a)
    a = X_test.columns[X_test.isna().any()].tolist()
    print(a)

    # Model training
    model2.fit(X_train, y_train['Income'].values.ravel())
    model.fit(X_train, y_train['Income'].values.ravel())

    # Model prediction
    y_pred_lr = model.predict(X_test)
    A_pred_lr = model.predict(A_test)

    y_pred_rr = model2.predict(X_test)
    A_pred_rr = model2.predict(A_test)

    rms = sqrt(mean_squared_error(b_test, A_pred_lr))
    print(rms)

    rms = sqrt(mean_squared_error(b_test, A_pred_rr))
    print(rms)

    df = pd.DataFrame({'Income1':y_pred_lr , 'Income2':y_pred_rr})

    income = df.mean(axis=1)
    
    # Create submission dataframe
    df_1 = pd.DataFrame({'Instance':instance.flatten(),'Total Yearly Income [EUR]': income})

    # Save dataframe to submission file as csv
    df_1.to_csv(r'tcd ml 2019-20 income prediction submission file.csv')

if __name__ == '__main__':
    main()