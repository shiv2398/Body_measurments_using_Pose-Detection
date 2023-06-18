import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
#from xgboost import XGBRegressor
import json
from configuration_files.model_utils import MODEL_CONFIG
class models:
    def __init__(self):
        self.model_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),'configuration_files/regression_model.json')

    def load_json(self):
        with open(self.model_file) as json_file:
            return json.load(json_file)

    def regression_models(self, model_name):
        data = self.load_json()
        model_data = data.get(model_name)
        
        if model_data:
            model_str = model_data.get('model')
            model_hyperparameters = model_data.get('hyperparameters', {})

            # Get hyperparameters from YAML configuration
            yaml_hyperparameters = MODEL_CONFIG.get(model_name, {})
            for hyperparameter, values in yaml_hyperparameters.items():
                # Use the first value as the default if multiple values are provided
                model_hyperparameters[hyperparameter] = values[0] if isinstance(values, list) else values
            try:
                # Evaluate the model string to instantiate the model
                model = eval(model_str)(**model_hyperparameters)
                return model, model_hyperparameters
            except NameError:
                print(f"Error: Model '{model_str}' is not defined.")
        else:
            print(f"Error: Model '{model_name}' is not found in the configuration file.")

        return None, {}



'''md=models()
model_name = 'Ridge Regression'
model, hyperparameters = md.regression_models(model_name)

# Output the extracted model and hyperparameters
print(model)  # LinearRegression()
print(hyperparameters)  # {'param1': 'value1', 'param2': 'value2'}
import numpy as np
x=np.array([1,2,3,4]).reshape(-1,1)
y=np.array([2,3,4,5]).reshape(-1,1)
model.fit(x,y)'''

