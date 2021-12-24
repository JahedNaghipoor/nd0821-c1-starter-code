"""
In this project, we will find customers who are likely to churn
Author: Jahed Naghipoor
Date: 20/10/2021
Pylint score: 8.29/10
"""
import os
import logging
import joblib

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


os.environ['QT_QPA_PLATFORM'] = 'offscreen'

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")


def import_data(path):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    try:
        # Trying to read file
        dataframe = pd.read_csv(path)
        logging.info(f'SUCCESS: file {path} loaded successfully')
    except FileNotFoundError as err:
        logging.error(f'ERROR: file {path} not found')
        raise err
    return dataframe


def perform_eda(dataframe):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    plotting_columns = [
        "Churn",
        "Customer_Age",
        "Marital_Status",
        "Total_Trans_Ct",
        "Heatmap"]

    img_eda_pth = "./images/eda/"
    if not os.path.exists(img_eda_pth):
        os.makedirs(img_eda_pth)
    for column in plotting_columns:

        plt.figure(figsize=(20, 10))

        if column == 'Churn':
            dataframe['Churn'].hist()

        elif column == 'Customer_Age':
            dataframe['Customer_Age'].hist()

        elif column == 'Marital_Status':
            dataframe.Marital_Status.value_counts('normalize').plot(kind='bar')

        elif column == 'Total_Trans_Ct':
            sns.distplot(dataframe['Total_Trans_Ct'])

        elif column == 'Heatmap':
            sns.heatmap(
                dataframe.corr(),
                annot=False,
                cmap='Dark2_r',
                linewidths=2)

        plt.savefig(os.path.join(img_eda_pth, f"{column}.jpg"))
        plt.close()


def encoder_helper(dataframe, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for category_name in category_lst:
        category_lst = []
        category_groups = dataframe.groupby(category_name).mean()["Churn"]
        for val in dataframe[category_name]:
            category_lst.append(category_groups.loc[val])
        dataframe[category_name + "_" + response] = category_lst
    return dataframe


def perform_feature_engineering(dataframe):
    '''
    input:
              df: pandas dataframe
              response: string of response name
              [optional argument that could be used for naming variables or index y column]
    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    try:
        assert isinstance(dataframe, pd.DataFrame)
    except AssertionError as err:
        logging.error(
            f'ERROR: Type to be {pd.DataFrame} but is {type(dataframe)}')
        raise err
# Input columns
    x_cols = [
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
        "Gender_Churn",
        "Education_Level_Churn",
        "Marital_Status_Churn",
        "Income_Category_Churn",
        "Card_Category_Churn"]

    X = pd.DataFrame()
    # Teature columns
    X[x_cols] = dataframe[x_cols]
    # Target column
    y = dataframe["Churn"]

    # Spliting the data to 70% train and 30% test
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    logging.info('SUCCESS: Data splitting finished.')
    logging.info(f'INFO: X_train size {x_train.shape}.')
    logging.info(f'INFO: X_test size {x_test.shape}.')
    logging.info(f'INFO: y_train size {y_train.shape}.')
    logging.info(f'INFO: y_test size {y_test.shape}.')
    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # scores
    print('random forest results')
    print('test results')
    test_results = classification_report(y_test, y_test_preds_rf)
    print(test_results)
    joblib.dump(
        test_results,
        './images/results/random_forest_test_results.joblib')
    print('train results')
    train_results = classification_report(y_train, y_train_preds_rf)
    print(train_results)
    joblib.dump(
        train_results,
        './images/results/random_forest_train_results.joblib')

    print('logistic regression results')
    print('test results')
    test_results = classification_report(y_test, y_test_preds_lr)
    print(test_results)
    joblib.dump(
        test_results,
        './images/results/logistic_regression_test_results.joblib')
    print('train results')
    train_results = classification_report(y_train, y_train_preds_lr)
    print(train_results)
    joblib.dump(
        train_results,
        './images/results/logistic_regression_train_results.joblib')


def feature_importance_plot(model, X, output_path):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X.shape[1]), names, rotation=90)
    # save plots
    output_path = './images/feat_imp_plot.jpeg'
    plt.savefig(output_path)
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''

    rfc = RandomForestClassifier(random_state=42)
    rfc.fit(X_train, y_train)
    y_train_preds_rf = rfc.predict(X_train)
    y_test_preds_rf = rfc.predict(X_test)

    lrc = LogisticRegression(max_iter=1000)
    lrc.fit(X_train, y_train)
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    result_pth = './images/results/'

    if not os.path.exists(result_pth):
        os.makedirs(result_pth)

    feature_importance_plot(rfc, X_train, './images/results/')

    model_pth = "models/"
    if not os.path.exists(model_pth):
        os.makedirs(model_pth)

    joblib.dump(rfc, "models/rfc_model.pkl")
    joblib.dump(lrc, "models/logistic_model.pkl")


if __name__ == "__main__":
    PATH = "./data/bank_data.csv"

    df = import_data(PATH)

    category_list = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    perform_eda(df)

    df = encoder_helper(df, category_list, 'Churn')

    X_train, X_test, y_train, y_test = perform_feature_engineering(df)

    train_models(X_train, X_test, y_train, y_test)
