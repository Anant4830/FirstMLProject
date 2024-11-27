import os
import sys

import mlflow
import mlflow.sklearn
import numpy as np
import pickle

from src.utils.utils import load_object
from src.exceptions.exception import customException
from src.logger.custom_logging import logging

from urllib.parse import urlparse
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelEvaluation:
    def __init__(self):
        logging.info("Evaluation started...!!")

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        logging.info("all evaluation metrices are captured....!!")
        return rmse, mae, r2
    
    def initiate_model_evaluation(self, train_array, test_array):
        try:
            X_test, y_test = (test_array[:, :-1], test_array[:, -1])

            model_path = os.path.join("artifacts", "model.pkl")
            model = load_object(model_path)

            #mlflow.set_registry_uri("")

            logging.info("madel has been registered-----")

            tracking_uri_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            print(tracking_uri_type_store)


            with mlflow.start_run():
                prediction = model.predict(X_test)
                (rmse, mae, r2) = self.eval_metrics(y_test, prediction)
               
          
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)
                print("After log:")

                # if tracking_uri_type_store != "file":
                #     print("in if block")
                #     mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
                # else:
                #     print("in else block")
                #     mlflow.sklearn.log_model(model, "model")
                # print("Last else")
        
        except Exception as e:
            logging.info("Evaluation complete...!!")
            raise customException(e, sys)