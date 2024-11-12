import pandas as pd
import numpy as np

from src.logger.custom_logging import logging
from src.exceptions.exception import customException

import os
import sys


from dataclasses import dataclass
from pathlib import Path

from src.utils.utils import save_object, evaluate_model

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting Dependent and Independent Variable')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'LinearRegression':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'ElasticNet':ElasticNet()
            }

            model_report:dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print("\n================================")
            logging.info(f'Model Report : {model_report}')

            #to get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
               list(model_report.values()).index(best_model_score)
            ]
        
            best_model = models[best_model_name]

            print(f'Best Model Found, model name : {best_model_name}',)
            print("\n===========================")
            logging.info(f'Best Model Found, Model Name : {best_model}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
        

        except Exception as e:
            logging.info("Inside model Trainer")
            raise customException(e, sys)