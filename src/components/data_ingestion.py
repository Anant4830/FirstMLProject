import pandas as pd
import numpy as np

from src.logger.custom_logging import logging
from src.exceptions.exception import customException

import os
import sys

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    raw_data_path:str=os.path.join("artifacts", "raw.csv")
    train_data_path:str=os.path.join("artifacts", "train.csv")
    test_data_path:str=os.path.join("artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started...!")
        try:
            # data = pd.read_csv(r"C:\Users\Admin\Downloads\AI-839\MLOps\projects2\FirstMLProject\data\cubic_zirconia.csv")
            #path_csv = os.path.abspath(os.path.join("..","..","data","cubic_zirconia.csv"))
            # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            # file_path = os.path.join(BASE_DIR, "data", "cubic_zirconia.csv")
            # # data = pd.read_csv(path_csv)
            data = pd.read_csv(r"/app/data/cubic_zirconia.csv")
            #data = pd.read_csv(file_path)
            logging.info("Reading my dataFrame")

            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)))
                        # os.path.dirname(os.path.join(self.ingestion_config.train_data_path)),
                        # os.path.dirname(os.path.join(self.ingestion_config.test_data_path)))
            
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("I have saved the raw dataset in the artifacts folder")

            logging.info("Here, I have performed train test split")
            train_data, test_data = train_test_split(data, test_size=0.25)
            logging.info("train test split completed")

            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("data ingestion part completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
   
        except Exception as e:
            logging.info("Inside")
            raise customException(e, sys)
        
# if __name__ == "__main__":
#     obj=DataIngestion()
#     obj.initiate_data_ingestion()
