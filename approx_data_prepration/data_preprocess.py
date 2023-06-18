import os 
import sys
import pandas as pd 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import circum_cal
class data_prep:
    """
        Preprocess the data by dropping rows with null values, shuffling the dataframe,
        and replacing zero values with column means.

        Returns:
            pd.DataFrame: Original data and preprocessed data.
    """
    def __init__(self,df):

        self.data=df

    def preprocess_data(self):
    
        df=self.data.copy()
        has_null = df.isnull().any()
        print("Has null values:\n", has_null)
        # Drop rows with null values
        df.dropna(inplace=True)

        #shuffling the df
        df = df.sample(frac=1).reset_index(drop=True)
        # Replace zero values with column means
        for col in df.columns:
            df[col] = df[col].replace(0, df[col].mean())
        return self.data,df


    
