
# Load the Pandas libraries with alias pd
import pandas as pd
# Load the different libraries needed from sklearn
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
# Load numpy libraries with alias np
import numpy as np
from scipy import stats
import math
from math import sqrt

def deal_with_nan(data):
    imp_mean = preprocessing.Imputer(missing_values=np.nan, strategy='most_frequent', axis=1)
    #data = imp_mean.fit(data)
    # Mean any rows which have any nan's
    data = pd.DataFrame(imp_mean.fit_transform(data))

    #for col in data.columns:
    #    data[col] = data[col].fillna(data[col].mean())
        #data[col].mean()
    #print(data.dtypes)
    return data

def main():
    # Read data from files
    training_data1 = pd.read_csv("tcdml1920-rec-click-pred--training(2).csv")
    training_data2 = pd.read_csv("tcdml1920-rec-click-pred--training(2).csv")
    training_data = training_data1.append(training_data2)
    test_data = pd.read_csv("tcdml1920-rec-click-pred--test.csv")

    split_data_at = training_data.shape[0]
    print(split_data_at)

    # Join training and test data
    appended = training_data.append(test_data)

    #Na
    appended = appended.replace("\N", np.nan)
    #Na
    appended = appended.replace("nA", np.nan)

    # drop set_clicked 
    appended = appended.drop(['set_clicked', 'recommendation_set_id'], axis=1)

    #'query_word_count','query_char_count','year_published','number_of_authors','abstract_word_count','abstract_char_count',

    # Clean Data
    #appended = appended[['organization_id','hour_request_received','rec_processing_time','number_of_recs_in_set','clicks','ctr','set_clicked','query_detected_language','query_identifier','abstract_detected_language','application_type','item_type','app_lang','algorithm_class','cbf_parser','search_title','search_keywords']]

    # Label encode the dataset columns that are not ints of training + test
    for column in appended.columns:
        if appended[column].dtype == type(object):
            le = preprocessing.LabelEncoder()
            appended[column] = le.fit_transform(appended[column])

    appended = deal_with_nan(appended)

    # Split data back into training and test seperately
    df_1 = pd.DataFrame()
    df_2 = pd.DataFrame()
    if appended.shape[0] > split_data_at: 
        df_1 = appended[:split_data_at]
        df_2 = appended[split_data_at:]

    train = df_1
    test = df_2

    y_train = pd.DataFrame({'set_clicked':training_data['set_clicked']})
    y_ans = pd.DataFrame({'recommendation_set_id':test_data['recommendation_set_id']})

    a = train.columns[train.isna().any()].tolist()
    print(a)
    a = test.columns[test.isna().any()].tolist()
    print(a) 

    toDrop = [] 

    for i in train:
        corr = train[i].corr(y_train['set_clicked'])
        if(corr <= 0.04):
            toDrop.append(i)
            #print(train.columns.values[i],'corr= ', corr)
        if(corr > 0.04):
            print(train.columns.values[i],'corr= ', corr)
    
    print(toDrop)
    train = train.drop(toDrop,axis=1)
    test = test.drop(toDrop,axis=1)
    # iterating the columns 
    # for col in train.columns: 
    #     print(col) 
    # iterating the columns 
    # for col in test.columns: 
    #     print(col) 

    model = RandomForestClassifier(n_estimators=10)
    model2 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    #RandomForestRegressor(n_estimators=10)

    model.fit(train, y_train)

    A = model.predict(test)

    df = pd.DataFrame({'set_clicked': A.flatten()})

    #df.loc[df.set_clicked_lr <= 10, 'set_clicked'] = '0'
    #df.loc[df.set_clicked_lr > 10, 'set_clicked'] = '1'
    
    # Create submission dataframe
    df_1 = pd.DataFrame({'recommendation_set_id':y_ans['recommendation_set_id'],'set_clicked': df['set_clicked']})

    # Save dataframe to submission file as csv
    df_1.to_csv(r'tcd ml 2019-20 rec prediction submission file.csv')

if __name__ == '__main__':
    main()