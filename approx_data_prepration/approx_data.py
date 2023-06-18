import os 
import sys
import cv2
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from approx_data_prepration.Distance_approx import circum_calculations
from pose_detection_model.pose_detetion import pose_detection
from approx_data_prepration.csv_refine import joints_file_preparation
from utils import DATA_CONFIG
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def approximation_data_formation(original_data_csv=None,
                                  image_data=None,
                                  csv_save:bool=True,
                                  test_mode=False,
                                  image_path:str=None):
    """
    Perform data approximation based on pose detection.

    Args:
        original_data_csv: Path to the original data CSV file.
        image_data: Path to the image data directory.
        csv_save: Specify whether to save the resulting CSV file.
        test_mode: Specify whether to run the function in test mode.
        image_path: Path to the image file for test mode.

    Returns:
        pd.DataFrame: Dataframe containing the approximated data.
    """
    # Rest of the function code
    pose_detection_obj = pose_detection()  # Instantiate the pose_detection class
    jfp_obj = joints_file_preparation()  # Instantiate the csv_and joint extraction class
    circum_data = []  # List to store circum dictionaries
    if test_mode:
        joints_dict = pose_detection_obj.pose_estimation(image_path=image_path,
                                                         pose_show=False,
                                                         extracted_pose=False,
                                                         return_joints=True,
                                                         save_csv=False)
        if joints_dict is None:
            circum_dict = {'shoulders': 0, 'hips': 0, 'waist': 0}
        else:
            circum_dict = circum_calculations(jfp_obj.extract_coordinates(joints_dict),image_path)
        return circum_dict
    data = pd.read_csv(original_data_csv)
    for participant in tqdm(data['Participant ID']):
        #print(participant)
        # image_path mapping
        image_data_dir = os.path.join(image_data, str(participant))
        image_folder = os.listdir(image_data_dir)
        image_path = os.path.join(image_data_dir, image_folder[0])
        img=cv2.imread(image_path)
        # pose_estimation
        joints_dict = pose_detection_obj.pose_estimation(image_path=image_path,
                                                         pose_show=False,
                                                         extracted_pose=False,
                                                         return_joints=True,
                                                         save_csv=False)
        if joints_dict is None:
            circum_dict = {'shoulders': 0, 'hips': 0, 'waist': 0}
        else:
            circum_dict = circum_calculations(jfp_obj.extract_coordinates(joints_dict),image_path)
        circum_dict['Participant ID'] = participant  # Add Participant ID to the circum_dict
        circum_data.append(circum_dict)
    df = pd.DataFrame(circum_data)  # Create a dataframe from the circum_data list
    
    df = df[['Participant ID', 'shoulders', 'hips', 'waist']]  # Rearrange the columns if needed

    if csv_save:
        folder_name = 'approximate_csv_data'
        current_dir = os.getcwd()
        folder_path = os.path.join(current_dir, folder_name)

        # Check if the folder already exists
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created folder '{folder_name}'")

        # Save the CSV file in the created folder
        file_path = os.path.join(folder_path, 'Approx_data(distances).csv')
        df.to_csv(file_path, index=False)
        print(f"Saved CSV file: {file_path}")
    return df

#df = approximation_data_formation('progress_work/Measurements.xlsx - Sheet1.csv','/media/sahitya/1674386C743850AB/AIMIIR assignments/Data')
