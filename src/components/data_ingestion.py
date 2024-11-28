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
        logging.info("Data Ingestion Started")
        try:
            data = pd.read_csv(r"/app/data/cubic_zirconia.csv")
            logging.info("Read Data")

            artifacts_dir = "/app/artifacts"

            if os.path.exists(artifacts_dir) and os.path.isdir(artifacts_dir):
                for item in os.listdir(artifacts_dir):
                    item_path = os.path.join(artifacts_dir, item)
                    os.unlink(item_path)
            
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Saved Data to artifacts")

            logging.info("Train Test Split")
            train_data, test_data = train_test_split(data, test_size=0.25)

            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Data Ingestion Completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
   
        except Exception as e:
            logging.info("Error in Data Ingestion")
            raise customException(e, sys)
        
# if __name__ == "__main__":
#     obj=DataIngestion()
#     obj.initiate_data_ingestion()
