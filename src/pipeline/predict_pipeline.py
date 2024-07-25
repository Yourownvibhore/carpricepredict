import sys
import os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            preprocessor_path="artifact/preprocessor.pkl"
            model_path="artifact/model.pkl"
            preprocessor=load_object(file_path=preprocessor_path)
            model=load_object(file_path=model_path)
            data_scaled=preprocessor.transform(features)
            model_prediction=model.predict(data_scaled)
            return model_prediction
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(
            self,
            year,
            kilometers,
            fuelType,
            transmission,
            Owner_Type,
            Mileage,
            Engine,
            Power,
            Seats,
            Brand,
            Model,
            region,
    ):
        self.year = year
        self.kilometers = kilometers
        self.fuelType = fuelType
        self.transmission = transmission
        self.Owner_Type = Owner_Type
        self.Mileage = Mileage
        self.Engine = Engine
        self.Power = Power
        self.Seats = Seats
        self.Brand = Brand
        self.Model = Model
        self.region = region
    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict={
                "Year":[self.year],
                "Kilometers_Driven":[self.kilometers],
                "Fuel_Type":[self.fuelType],
                "Transmission":[self.transmission],
                "Owner_Type":[self.Owner_Type],
                "Mileage(kmpl)":[self.Mileage],
                "Engine(cc)":[self.Engine],
                "Power(bhp)":[self.Power],
                "Seats":[self.Seats],
                "Brand":[self.Brand],
                "Model":[self.Model],
                "Region":[self.region],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)