import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

import os
import sys

from src.exceptions.exception import customException
from src.logger.custom_logging import logging
from src.utils.utils import load_object, save_object

# from mapie.regression import MapieRegressor
from pyod.models.mcd import MCD
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.spatial.distance import mahalanobis
from scipy.stats import probplot
from dataclasses import dataclass
from pathlib import Path



@dataclass
class DataReportConfig:
    shap_summary_plot_file_path:str = os.path.join('static', '1_shap_summary_plot.png')
    shap_force_plot_file_path:str = os.path.join('static', '2_shap_force_plot.png')
    calibration_plot_file_path:str = os.path.join('static', '3_calibration.png')

class DataReportGen:
    def __init__(self):
        self.data_report_config=DataReportConfig()
        self.out = {}

    def __inference_with_explainability_and_trust(self,
    instance, model, X_train, X_test, pyod_model, conformal_interval, explainer
    ):
        # Model Prediction
        print(instance.reshape(1, -1).shape)
        prediction = model.predict(instance.reshape(1, -1))[0]

        # Explainability: Compute SHAP explanation
        shap_values = explainer.shap_values(instance)
        explanation = shap_values

        # Conformal Prediction: Extract interval for this instance
        instance_index = np.where((X_test == instance).all(axis=1))[0]
        if len(instance_index) == 0:
            raise ValueError("Instance not found in test set for conformal prediction.")
        conformal_bounds = conformal_interval[instance_index[0]]

        # Trust Score: Mahalanobis Distance
        cov_model = EmpiricalCovariance()
        cov_model.fit(X_train)
        mean = np.mean(X_train, axis=0)
        mahal_dist = mahalanobis(instance, mean, cov_model.precision_)
        mahalanobis_confidence = np.exp(-mahal_dist)  # Convert distance to confidence

        # Trust Score: PyOD
        pyod_confidence = pyod_model.decision_function(instance.reshape(1, -1))[0]

        # Combined Trust Score
        combined_confidence = 0.5 * mahalanobis_confidence + 0.5 * pyod_confidence

        return {
            "prediction": prediction,
            "explanation": explanation,
            "conformal_bounds": conformal_bounds,
            "mahalanobis_confidence": mahalanobis_confidence,
            "pyod_confidence": pyod_confidence,
            "combined_confidence": combined_confidence,
        }


    def get_data_analysis_aggregate(self, train_array, test_array):
        try:
            logging.info("Starting Report Gen")
            # df_train = pd.read_csv(train_data_path)
            # df_test = pd.read_csv(test_data_path)
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            model_path = os.path.join("artifacts", "model.pkl")
            model = load_object(model_path)
            print("Loaded Model")
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print(f"MSE: {mse}, MAE: {mae}, R2: {r2}")
            self.out["mse"] = mse

        except Exception as e:
            print("Error", e)

    def get_data_analysis_inference(self):
        try:
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            
            df_train = pd.read_csv(os.path.join("artifacts", "train.csv"))
            df_test = pd.read_csv(os.path.join("artifacts", "test.csv"))

            y_train = df_train['price']
            y_test = df_test['price']

            X_train = df_train.drop(columns=['price'])
            X_test = df_test.drop(columns=['price'])

            X_train = pd.DataFrame(preprocessor.transform(X_train), columns=preprocessor.get_feature_names_out())
            X_test = pd.DataFrame(preprocessor.transform(X_test), columns=preprocessor.get_feature_names_out())
            
            explainer = shap.LinearExplainer(model, X_train)
            shap_values = explainer(X_test)
            plt.figure()
            shap.summary_plot(shap_values, X_test, show=False)
            plt.savefig("static/1_shap_summary_plot.png", bbox_inches='tight', dpi=300)  # Save with high resolution
            plt.close()

            index = 0  
            plt.figure()
            shap.force_plot(
                explainer.expected_value,  # Expected value (base value)
                shap_values[index].values,  # SHAP values for the instance
                X_test.iloc[index, :],  # Feature values for the instance
                matplotlib=True,  
                show=False
            )
            plt.savefig("static/2_shap_force_plot.png", bbox_inches='tight', dpi=300) 
            plt.close()

            y_pred = model.predict(X_test)

            residuals = y_test - y_pred
            plt.figure(figsize=(8, 6))
            probplot(residuals, dist="norm", plot=plt)
            plt.title("QQ Plot for Residuals")
            plt.savefig("static/3_calibration.png", bbox_inches='tight', dpi=300)

            noise = np.random.normal(0, 0.1, X_test.shape)
            X_test_noisy = X_test + noise
            y_pred_noisy = model.predict(X_test_noisy)
            mse_noisy = mean_squared_error(y_test, y_pred_noisy)

            X_test_missing = X_test.copy()
            missing_rate = 0.1  # 10% missing data
            missing_mask = np.random.rand(*X_test_missing.shape) < missing_rate
            X_test_missing[missing_mask] = np.nan
            X_test_missing.fillna(X_test.mean(), inplace=True)  # Simple imputation
            y_pred_missing = model.predict(X_test_missing)
            mse_missing = mean_squared_error(y_test, y_pred_missing)

            
            self.out["mse_missing"] = mse_missing
            self.out["mse_noisy"] = mse_noisy
            
            pyod_model = MCD()
            pyod_model.fit(X_train)

            # Conformal Prediction: Quantile Regression for Prediction Intervals
            lower_quantile = np.quantile(y_test - model.predict(X_test), 0.05)
            upper_quantile = np.quantile(y_test - model.predict(X_test), 0.95)
            conformal_interval = np.array([[pred + lower_quantile, pred + upper_quantile] for pred in model.predict(X_test)])
            #Step 2: Inference with Explainability and Trust Scores
            # Test on a single instance
            index = 0
            instance = X_test.iloc[index, :].values
            print("instance:")
            print(instance.shape)

            result = self.__inference_with_explainability_and_trust(
                instance=instance,
                model=model,
                X_train=X_train,
                X_test=X_test,
                pyod_model=pyod_model,
                conformal_interval=conformal_interval,
                explainer=explainer,
            )            

            self.out.update(result)
            print(self.out)
            save_object(os.path.join("artifacts", "report.pkl"), self.out)
            

        except Exception as e:
            #
            print("Error")
            raise customException(e, sys)