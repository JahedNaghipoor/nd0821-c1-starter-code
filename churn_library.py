"""
In this project, we will find customers who are likely to churn
Author: Jahed Naghipoor
Date: 20/10/2021
Pylint score: 9.10/10
"""
import os
import logging
import joblib

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

IMAGE_EDA_PATH = "./images/eda/"
MODEL_PATH = "models/"
IMAGE_RESULT_PATH = './images/results'

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
            dataframe: pandas dataframe
    '''
    try:
        # Trying to read file
        dataframe = pd.read_csv(path)
        logging.info('SUCCESS: file %s loaded successfully', path)
    except FileNotFoundError as err:
        logging.error('ERROR: file %s not found', path)
        raise err
    return dataframe


def perform_eda(dataframe):
    '''
    perform eda on df and save figures to images folder
    input:
            dataframe: pandas dataframe

    output:
            None
    '''
    plotting_columns = [
        "Churn",
        "Customer_Age",
        "Marital_Status",
        "Total_Trans_Ct",
        "Heatmap"]

    if not os.path.exists(IMAGE_EDA_PATH):
        os.makedirs(IMAGE_EDA_PATH)
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

        plt.savefig(os.path.join(IMAGE_EDA_PATH, f"{column}.jpg"))
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
            dataframe: pandas dataframe with new columns
    '''
    for category_name in category_lst:
        category_lst = []
        category_groups = dataframe.groupby(category_name).mean()["Churn"]
        for val in dataframe[category_name]:
            category_lst.append(category_groups.loc[val])
        dataframe[category_name + "_" + response] = category_lst
    return dataframe


def perform_feature_engineering(dataframe, response):
    '''
    input:
              dataframe: pandas dataframe
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
    keep_cols = [
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

    features = pd.DataFrame()
    # Teature columns
    features[keep_cols] = dataframe[keep_cols]
    # Target column
    label = dataframe[response]

    # Spliting the data to 70% train and 30% test
    x_train, x_test, y_train, y_test = train_test_split(
        features, label, test_size=0.3, random_state=42)
    logging.info('SUCCESS: Data splitting finished.')
    logging.info('INFO: X_train size: %s', str(x_train.shape))
    logging.info('INFO: X_test size: %s', str(x_test.shape))
    logging.info('INFO: y_train size: %s', str(y_train.shape))
    logging.info('INFO: y_test size: %s', str(y_test.shape))
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


def feature_importance_plot(model, features, output_path):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            features: pandas dataframe of X values
            output_path: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [features.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(features.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(features.shape[1]), names, rotation=90)
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
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

    cv_rfc.fit(X_train, y_train)
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    lrc = LogisticRegression()
    lrc.fit(X_train, y_train)
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    plot_roc_curve(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=axis,
        alpha=0.8)
    lrc_plot.plot(ax=axis, alpha=0.8)
    plt.savefig('./images/ROC_curves.png')

    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    joblib.dump(
        cv_rfc.best_estimator_,
        os.path.join(
            MODEL_PATH,
            "/rfc_model.pkl"))
    joblib.dump(lrc, os.path.join(MODEL_PATH, "/logistic_model.pkl"))


if __name__ == "__main__":
    PATH = "./data/bank_data.csv"

    df = import_data(PATH)
    RESPONSE = 'Churn'
    category_list = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

    df[RESPONSE] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    perform_eda(df)

    df = encoder_helper(df, category_list, RESPONSE)

    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df, RESPONSE)

    train_models(X_train, X_test, y_train, y_test)

    rfc_model = joblib.load(os.path.join(MODEL_PATH, '/rfc_model.pkl'))
    feature_importance_plot(rfc_model, X_train, IMAGE_RESULT_PATH)
