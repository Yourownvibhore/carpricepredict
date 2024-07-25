import os
import dill
import sys
from src.exception import CustomException
from src.logger import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.utils import evaluate_model, save_object
from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

@dataclass
class ModelTrainerConfig:
    model_pkl_file_path=os.path.join('artifact','model.pkl')

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer=ModelTrainerConfig()
    
    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info("splitting the data into X_train,y_train,X_test,y_test")
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1],
            )
            logging.info("splitting completed")
            logging.info("Initiating model training")
            modelss={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "XGBRegressor":XGBRegressor(),
                "CatBoosting Regressor":CatBoostRegressor(),
                "AdaBoost Regressor":AdaBoostRegressor(),
                "KNeighbors Regressor":KNeighborsRegressor()
            }

            logging.info("Evaluating the models")
            model_report:dict=evaluate_model(modelss,X_train,y_train,X_test,y_test)
            logging.info(f"model_report: {model_report}")
            best_model_score=max(model_report.values())
            logging.info(f"best_model_score: {best_model_score}")
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            logging.info(f"best_model_name: {best_model_name}")
            best_model=modelss[best_model_name]
            logging.info(f"best_model: {best_model}")

            if best_model_score<0.6:
                raise CustomException("model score is less than 0.6, no best model found")
            logging.info("found the best model")
            save_object(
                self.model_trainer.model_pkl_file_path,
                best_model
            )
            logging.info("model saved successfully")
            predicted=best_model.predict(X_test)
            logging.info(f"predicted value: {predicted}")
            r2_square=r2_score(y_test,predicted)
            logging.info(f"r2 score: {r2_square}")
            return r2_square
        except Exception as e:
            raise CustomException(e,sys)