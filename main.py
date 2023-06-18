import sys
import os
import pandas as pd 
import numpy as np 
import argparse
import logging
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from regression_model import regression_
import training.trainingfile as trainingfile
from approx_data_prepration.approx_data import approximation_data_formation
from utils import circum_cal
from approx_data_prepration.data_preprocess import data_prep
import pickle
from regression_model import metrics
from tqdm import tqdm
import random
def configure_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    return logger



def save_model(model, file_name):
    file_name = file_name + '.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump(model, f)
    print(f'The model has been saved to: {file_name}')

def cm_to_inches(cm):
    inches = cm / 2.54
    return inches


def distance_to_circum(df):
    for col in df.columns:
        if col=='Participant ID':
            pass
        else:
            df[col]=[circum_cal(distance) for distance in df[col].tolist()]
    return df

def shape_changer(x):
    return np.array(x).reshape(-1,1)

def Approximation_data_formation(logger,image_dir,odf_csv,circum_convert=False):
       # Perform data approximation logic here
    logger.info('Data approximation mode')
    # Add your data approximation code here
    approx_df=approximation_data_formation(image_dir,odf_csv)
    df_prepr_obj=data_prep(approx_df)
    #data_preprocessing
    _,df=df_prepr_obj.preprocess_data()
    #changing the joint distanes into the circumferences 
    
    df_c=distance_to_circum(df)
    return df,df_c
    
def training_model(logger, orig_df, approx_df, columns_to_train=None, separate_model_train=False, 
                   model_name=None, model_list=None, model_save=False):
    
    logger.info('Regression training mode')
    logger.info(f'Seprate_Model Training: {separate_model_train}')
    training_obj = trainingfile.training()
    model_dict = {}
    model_save_info = {}
    default_model = 'Linear Regression'
  
    if separate_model_train:
        folder_name = f'saved_models/single_model/{model_name}{str(random.randint(1, 10000))}_separate_models'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        else:
            print(f'Error: The "{folder_name}" folder already exists. Please choose a different folder name.')
            return

        for col in columns_to_train:
            # Data preparation
            orig_ar = shape_changer(orig_df[col])
            approx_ar = shape_changer(approx_df[col])
        
            # Model training for the column
            model = training_obj.single_model(model_name, approx_ar, orig_ar)
            # Save the model to a pickle file
            if model_save:
                md_n = f'{folder_name}/{model_name}_{col}'
                save_model(model, md_n)
                model_save_info[md_n] = model
            model_dict[col] = model
    else:
        folder_name = f'saved_models/multiple_model/{model_list[0] + model_list[-1]}/{str(random.randint(1, 10000))}_multiple_models'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        else:
            print(f'Error: The "{folder_name}" folder already exists. Please choose a different folder name.')
            return
        if model_list is not None and len(model_list) != len(columns_to_train):
            while len(model_list) < len(columns_to_train):
                model_list.append(default_model)

        for col, model_name in tqdm(zip(columns_to_train, model_list)):
            # Data preparation
            orig_ar = shape_changer(orig_df[col])
            approx_ar = shape_changer(approx_df[col])

            # Model training for the column
            model = training_obj.single_model(model_name, approx_ar, orig_ar)
            model_dict[col] = model

            # Save the model to a pickle file
            if model_save:
                md_n = f'{folder_name}/{model_name}_{col}'
                save_model(model, md_n)
                model_save_info[md_n] = model
            model_dict[col] = model

    return model_dict



def model_testing(model, x_test, y_test, metric=True, inference=False):
    if isinstance(x_test,list):
        x_test=np.array(x_test)
    if x_test.ndim != 2 or x_test.shape[1] != 1:
        x_test = np.array(x_test).reshape(-1, 1)
    
    predict = model.predict(x_test).reshape(-1)
    y_test = np.array(y_test).reshape(-1)
    if metric:
        metric_dict=metrics.calculate_regression_metrics(predict, y_test, inference)
        return metric_dict, predict
    else:
        return predict

def new_data_testing(logger,image_path,single_model):
    logger.info('Testing mode')
    import os
    import pickle

    circum_dict=approximation_data_formation(test_mode=True,image_path=image_path)
    for key, value in circum_dict.items():
        circum_dict[key] = circum_cal(value)

    model_dict = {}

    if single_model:
        model_path = 'saved_models/single_model/Decision Tree Regression672_separate_models'

        with open(model_path+'/Decision Tree Regression_ChestGirth(cm).pkl', 'rb') as file:
                 model_dict['shoulders']= pickle.load(file)
        with open(model_path+'/Decision Tree Regression_HipsGirth(cm).pkl', 'rb') as file:
                 model_dict['hips']= pickle.load(file)
        with open(model_path+'/Decision Tree Regression_WaistGirth(cm).pkl', 'rb') as file:
                 model_dict['waist']= pickle.load(file)
    else:
        
        model_path = 'saved_models/multiple_model/Decision Tree RegressionGradient Boosting Regression/1073_multiple_models'

        with open(model_path+'/Decision Tree Regression_ChestGirth(cm).pkl', 'rb') as file:
                 model_dict['shoulders']= pickle.load(file)
        with open(model_path+'/ElasticNet Regression_HipsGirth(cm).pkl', 'rb') as file:
                 model_dict['hips']= pickle.load(file)
        with open(model_path+'/Gradient Boosting Regression_WaistGirth(cm).pkl', 'rb') as file:
                 model_dict['waist']= pickle.load(file)

    for (key,value),(_,value1) in zip(circum_dict.items(),model_dict.items()):
         circum_dict[key]=model_dict[key].predict(np.array(value).reshape(-1,1))
    return circum_dict
                
        


def main():
    logger = configure_logging()
    # Create the main parser
    parser = argparse.ArgumentParser(description='Body Measurements Using Pose Detection')
    subparsers = parser.add_subparsers(dest='mode', help='Choose mode: data_approximation, regression_training, or testing')

    # Create the parser for the data_approximation mode
    data_approx_parser = subparsers.add_parser('data_approximation', help='Perform data approximation')
    data_approx_parser.add_argument('--csv_path', help='Specify the CSV path')
    data_approx_parser.add_argument('--image_path', help='Specify the image path')
    #data_approx_parser.add_argument('--save_csv',help='Make this true if you want to save the csv')

    # Create the parser for the regression_training mode
    regression_training_parser = subparsers.add_parser('regression_training', help='Perform regression model training')
    regression_training_parser.add_argument('--measurements_csv_path', help='Specify the CSV path')
    regression_training_parser.add_argument('--separate_model_train', type=bool, help='One model for different columns')
    regression_training_parser.add_argument('--model_name', help='Specify the model name')
    regression_training_parser.add_argument('--model_name1', help='for multiple_model')
    regression_training_parser.add_argument('--model_name2', help='for multiple_model')
    regression_training_parser.add_argument('--model_name3', help='for multiple_model')
    regression_training_parser.add_argument('--save_model', type=bool, help='Specify whether to save the model')

    # Create the parser for the testing mode
    testing_parser = subparsers.add_parser('testing', help='Perform testing with new files')
    testing_parser.add_argument('--image_path', help='Specify the test data path')

    columns_to_train = ['ChestGirth(cm)', 'HipsGirth(cm)', 'WaistGirth(cm)']

    # Parse the command-line arguments
    args = parser.parse_args()

    output = {}  # Dictionary to store the output
    column_mapping = {
            'shoulders': 'ChestGirth(cm)',
            'hips': 'HipsGirth(cm)',
            'waist': 'WaistGirth(cm)'
        }
    if args.mode == 'data_approximation':
        circumference_data, data = Approximation_data_formation(logger, args.csv_path, args.image_path)
        output['data_approximation'] = {'circumference_data': circumference_data, 'data': data}
        print('Approximated data formation complete ---')

    elif args.mode == 'regression_training':
        approx_csv_path = os.path.join('approximate_csv_data', 'Approx_data(distances).csv')
        if not os.path.exists(approx_csv_path):
            print('Approximation data CSV file does not exist. Please provide the approximation data first.')
            return

        approx_df = pd.read_csv(approx_csv_path)

        # Create the model_list as a list of strings
        model_list = [args.model_name1, args.model_name2, args.model_name3]
        print(model_list)
        column_mapping = {
            'shoulders': 'ChestGirth(cm)',
            'hips': 'HipsGirth(cm)',
            'waist': 'WaistGirth(cm)'
        }

        approx_data = approx_df.rename(columns=column_mapping)

        orig_data = pd.read_csv(args.measurements_csv_path)
        print('Training Models---')
        trained_models = training_model(logger, orig_data, approx_data, columns_to_train,
                                        args.separate_model_train, args.model_name, model_list,
                                        args.save_model)

        predicted_data = pd.DataFrame(approx_data['Participant ID'])
        predicted_mse = pd.DataFrame()
        print('Calculating Metrics---')
        for col in columns_to_train:
            met, predicted_data[col] = model_testing(trained_models[col], approx_data[col].tolist(),
                                                    orig_data[col].tolist())
            predicted_mse[col] = pd.Series(met)

        print('Saving results--')

        # Split the string into a list of model names

        if args.separate_model_train:
            name = f'results/single_model/{model_list[0]}+separate_model_result'
        else:
            name = f'results/multiple_model/{model_list[0]+model_list[1]}+multiple_model{model_list[-1]}'

        result_folder = name if args.separate_model_train else name
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        else:
            print(f'Error: The "{result_folder}" folder already exists. Please choose a different folder name.')
            return

        # Save metrics as CSV
        metrics_path = os.path.join(result_folder, 'metrics.csv')
        predicted_mse.to_csv(metrics_path, index=False)

        # Save predicted data as CSV with model names in columns
        predicted_data_path = os.path.join(result_folder, 'predicted_data.csv')
        predicted_data.to_csv(predicted_data_path, index=False)

        output['regression_training'] = {'trained_models': trained_models, 'predicted_data': predicted_data,
                                         'predicted_mse': predicted_mse}

    elif args.mode == 'testing':
        testing_results=new_data_testing(logger, args.image_path,single_model=True)
        #output['testing'] = {'model_path': args.model_path, 'test_data_path': args.test_data_path}
        print('\033[1m\033[36m\033[4m \nOutput(in cm)\033[0m\n')
        for key,value in testing_results.items():
             print(f"\n{column_mapping[key]}  :  {float(value)}\n")
        print('\033[1m\033[36m\033[4m \nOutput(in inches)\033[0m\n')
        for key,value in testing_results.items():
             print(f"\n{column_mapping[key].split('(')[0]}  :  {float(cm_to_inches(value))}\n")
        
    else:
        logger.error('Invalid mode. Please choose data_approximation, regression_training, or testing.')

    
if __name__ == "__main__":
    main()




            


