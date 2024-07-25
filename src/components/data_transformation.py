# import os
# import sys
# from src.exception import CustomException
# from src.logger import logging
# import numpy as np
# import pandas as pd
# from dataclasses import dataclass
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder,StandardScaler,OrdinalEncoder

# from src.utils import save_object


# @dataclass
# class DataTransformationConfig:
#     preprocessor_pkl_file_path=os.path.join('artifact','preprocessor.pkl')

# class DataTransformation:
#     def __init__(self) -> None:
#         self.data_transformation_config=DataTransformationConfig()
    
#     def get_data_transformation_object(self):
#         try:
#             logging.info("Initiating data transformation")
#             numerical_columns=['Year', 'Kilometers_Driven', 'Mileage(kmpl)', 'Engine(cc)','Power(bhp)', 'Seats']
#             categorical_columns=['Fuel_Type', 'Transmission', 'Owner_Type', 'Brand', 'Model', 'Region']

#             logging.info("Data transformation initiated")

#             num_pipeline=Pipeline([
#                 ('imputer',SimpleImputer(strategy='median')),
#                 ('std_scaler',StandardScaler(with_mean=False))
#             ])
#             cat_pipeline=Pipeline([
#                 ('imputer',SimpleImputer(strategy='most_frequent')),
#                 ('one_hot_encoder',OneHotEncoder(handle_unknown='ignore')),
#                 ('std_scaler',StandardScaler(with_mean=False))
#             ])
#             logging.info(f"numerical_columns: {numerical_columns}")
#             logging.info(f"categorical_columns: {categorical_columns}")

#             preprocessor=ColumnTransformer([
#                 ('num',num_pipeline,numerical_columns),
#                 ('cat',cat_pipeline,categorical_columns),
#             ])
#             logging.info("Data transformation completed")
#             return preprocessor
        
#         except Exception as e:
#             raise CustomException(e,sys)


#     def initiate_data_transformation(self,train_path,test_path):
#         try:
#             train_df=pd.read_csv(train_path)
#             test_df=pd.read_csv(test_path)
#             logging.info("Read the data from the train.csv and test.csv file")
#             preprocessor_obj=self.get_data_transformation_object()
#             target_column='Price'
#             input_feature_train=train_df.drop(columns=[target_column],axis=1)
#             target_feature_train=train_df[target_column]

#             input_feature_test=test_df.drop(columns=[target_column],axis=1)
#             target_feature_test=test_df[target_column]

#             logging.info("fit the preprocessor object on the train data")
#             input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train)
#             logging.info("transform test data")
#             input_feature_test_arr=preprocessor_obj.transform(input_feature_test)
#             logging.info("Data transformation dsjndkjs")
#             logging.info(f"input_feature_train_arr: {input_feature_train_arr[0]}")
#             logging.info(f"input_feature_test_arr: {input_feature_test_arr[0]}")
#             train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train)]
#             test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test)]
#             logging.info("Data transformation completed")
#             logging.info("Saving the preprocessor object")

#             save_object(
#                 self.data_transformation_config.preprocessor_pkl_file_path,
#                 preprocessor_obj
#             )
#             return(
#                 train_arr,
#                 test_arr
#             )
#         except Exception as e:
#             raise CustomException(e,sys)

import os
import sys
from src.exception import CustomException
from src.logger import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,OrdinalEncoder

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_pkl_file_path=os.path.join('artifact','preprocessor.pkl')

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformation_object(self):
        try:
            logging.info("Initiating data transformation")
            numerical_columns=['Year', 'Kilometers_Driven', 'Mileage(kmpl)', 'Engine(cc)','Power(bhp)', 'Seats']
            categorical_columns_ordinal1=['Fuel_Type','Transmission','Owner_Type']
            categorical_columns_nominal=['Region']
            categorical_columns_ordinal2=['Brand','Model']
            target_column=['Price']

            logging.info("Data transformation initiated")

            num_pipeline=Pipeline([
                ('imputer',SimpleImputer(strategy='median')),
                ('std_scaler',StandardScaler(with_mean=False))
            ])
            cat_nominal_pipeline=Pipeline([
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder',OneHotEncoder(handle_unknown='ignore')),
                ('std_scaler',StandardScaler(with_mean=False))
            ])
            cat_ordinal1_pipeline=Pipeline([
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinal_encoder',OrdinalEncoder(categories=[['LPG','CNG','Petrol','Diesel'],['Manual','Automatic'],['Fourth & Above','Third','Second','First']])),
                ('std_scaler',StandardScaler(with_mean=False))
            ])
            cat_ordinal2_pipeline=Pipeline([
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinal_encoder',OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
                ('std_scaler',StandardScaler(with_mean=False))
            ])
            logging.info(f"numerical_columns: {numerical_columns}")
            logging.info(f"categorical_columns_ordinal1: {categorical_columns_ordinal1}")
            logging.info(f"categorical_columns_nominal: {categorical_columns_nominal}")
            logging.info(f"categorical_columns_ordinal2: {categorical_columns_ordinal2}")

            preprocessor=ColumnTransformer([
                ('num',num_pipeline,numerical_columns),
                ('cat_nominal',cat_nominal_pipeline,categorical_columns_nominal),
                ('cat_ordinal1',cat_ordinal1_pipeline,categorical_columns_ordinal1),
                ('cat_ordinal2',cat_ordinal2_pipeline,categorical_columns_ordinal2),
            ])
            logging.info("Data transformation completed")
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read the data from the train.csv and test.csv file")
            preprocessor_obj=self.get_data_transformation_object()
            target_column='Price'
            input_feature_train=train_df.drop(columns=[target_column],axis=1)
            target_feature_train=train_df[target_column]

            input_feature_test=test_df.drop(columns=[target_column],axis=1)
            target_feature_test=test_df[target_column]

            logging.info("fit the preprocessor object on the train data")
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train)
            logging.info("transform test data")
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test)
            
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test)]
            logging.info("Data transformation completed")
            logging.info("Saving the preprocessor object")

            save_object(
                self.data_transformation_config.preprocessor_pkl_file_path,
                preprocessor_obj
            )
            return(
                train_arr,
                test_arr
            )
        except Exception as e:
            raise CustomException(e,sys)