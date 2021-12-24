"""
In this file, we will test some functionalities of churn_library.py
Author: Jahed Naghipoor
Date: 20/10/2021
Pylint score: 9.24/10
"""
import os
import logging
import churn_library as chlib

logging.basicConfig(
    filename='./logs/churn_library_test.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")

def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        dataframe = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    return dataframe    


def test_eda(perform_eda, dataframe):
    '''
    test perform eda function
    '''
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    try:
        perform_eda(dataframe)
        path = "./images/eda"
    except AssertionError as err:
        logging.error("Error in perform_eda function")
        
        raise err

    # Checking if the list is empty or not
    try:
        # Getting the list of directories
        dir_val = os.listdir(path)
        assert len(dir_val) > 0
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.warning("Testing perform_eda function:"
                        "The image has not been saved in the eda folder.")
        raise err


def test_encoder_helper(encoder_helper, dataframe):
    '''
    test encoder helper
    '''
    categorical_columns = ['Gender', 'Education_Level', 'Marital_Status',
                   'Income_Category', 'Card_Category'
                   ]

    dataframe = encoder_helper(dataframe, categorical_columns, 'Churn')

    try:
        for column in categorical_columns:
            assert column in dataframe.columns
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The dataframe appears to be missing the "
            "transformed categorical columns")
        return err

    return dataframe


def test_perform_feature_engineering(perform_feature_engineering, dataframe):
    '''
    test perform_feature_engineering
    '''
    X_train, X_test, y_train, y_test = perform_feature_engineering(dataframe)
    try:
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: "
                      "The four objects that should be returned were not.")
        raise err

    return X_train, X_test, y_train, y_test


def test_train_models(train_models):
    '''
    test train_models
    '''
    train_models(X_train, X_test, y_train, y_test)
    path = "./images/results/"
    try:
        # Getting the list of directories
        dir_val = os.listdir(path)
        assert len(dir_val) > 0
    except FileNotFoundError as err:
        logging.error("Testing train_models: Results image files not found")
        raise err

    path = "./models/"
    try:
        # Getting the list of directories
        dir_val = os.listdir(path)
        assert len(dir_val) > 0
        logging.info("Testing train_models: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing train_models: Model files not found")
        raise err


if __name__ == "__main__":

    dataframe_import = test_import(chlib.import_data)
    test_eda(chlib.perform_eda, dataframe_import)
    dataframe_helper = test_encoder_helper(chlib.encoder_helper, dataframe_import)
    X_train, X_test, y_train, y_test = test_perform_feature_engineering(
        chlib.perform_feature_engineering, dataframe_helper)
    test_train_models(chlib.train_models)
    