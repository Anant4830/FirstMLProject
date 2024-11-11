import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.logger.custom_logging import logging
from src.exceptions.exception import customException

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise customException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            
            #Train model
            model.fir(X_train, y_train)

            #predict testing data
            y_test_pred = model.predict(X_test)

            #evaluate
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(model.keys())[i]] = test_model_score

        return report

        
    except Exception as e:
        logging.info("Exception occured during model training")
        raise customException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise customException(e, sys)
