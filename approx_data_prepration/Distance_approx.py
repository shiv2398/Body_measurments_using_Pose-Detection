import math
import csv
import pandas as pd 
import math
import os 
import sys
import cv2
from scipy.special import ellipe
from utils import *
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



def get_image_dimensions(image_path):
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    return height, width

def convert_normalized_to_pixel(normalized_coordinates,image_path):
    image_height,image_width=get_image_dimensions(image_path)
    pixel_coordinates = []
    for coord in normalized_coordinates:
        x = int(coord[0] * image_width)
        y = int(coord[1] * image_height)
        # z = coord[2]  # Z-coordinate remains unchanged
        pixel_coordinates.append((x, y))
    return pixel_coordinates


def extract_shoulder_hip_coordinates(joint_dict):
    shoulder_coordinates = []
    hip_coordinates = []

    for joint_name, joint_data in joint_dict.items():
        if "shoulder" in joint_name.lower():
            shoulder_coordinates.append((joint_data['X'], joint_data['Y']))
        elif "hip" in joint_name.lower():
            hip_coordinates.append((joint_data['X'], joint_data['Y']))
    #print(hip_coordinates)
    return shoulder_coordinates, hip_coordinates
    



def find_center_point(start_point, end_point):
    x1, y1 = start_point
    x2, y2 = end_point
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    center_point = (center_x, center_y)
    return center_point

def circum_calculations(joint_coordinates:dict=None,image_path:str=None,
                        csv_extraction:bool=False,
                        csv_path:str=None,
                        inference:bool=True)->dict:
    # Example usage:
    shoulder_coordinates = []
    #average_shoulder_width = 40   Example average shoulder width in centimeters
    hip_coordinates=[]
    hip_sh_waist_coor={}
    waist_coordinates=[None]*2
    shoulder_coordinates = []
    #below function is just to save the data into a csv file like the coordinates of joints 
    if csv_extraction:
        if os.path.exists(csv_path):
            # Read shoulder and hip coordinates from CSV file
            with open(csv_path, 'r') as csvfile:
                csvreader = csv.DictReader(csvfile)
                for row in csvreader:
                    joint = row['Joint']
                    x, y, z = float(row['X']), float(row['Y']), float(row['Z'])
                    if joint == 'right_shoulder' or joint == 'left_shoulder':
                        shoulder_coordinates.append((x, y))
                    if joint == 'right_hip' or joint == 'left_hip':
                        hip_coordinates.append((x, y))
        else:
            print('Path Not specified')
    #waist_coordinates    # Assuming waist coordinates are same as shoulder coordinates
    else:
        shoulder_coordinates, hip_coordinates=extract_shoulder_hip_coordinates(joint_coordinates)
    #waist coordinates
    waist_coordinates[0]=find_center_point(hip_coordinates[0],shoulder_coordinates[0])
    waist_coordinates[1]=find_center_point(hip_coordinates[1],shoulder_coordinates[1])
   
    hip_sh_waist_coor = {
        'shoulders': shoulder_coordinates,
        'hips': hip_coordinates,
        'waist':waist_coordinates
    }
    
    for name,coordinate in hip_sh_waist_coor.items():
        hip_sh_waist_coor[name]=convert_normalized_to_pixel(coordinate,image_path)
    
    #calculating the pixel distance between two joints 
    circum_map={}
    for name,coordinate in hip_sh_waist_coor.items():
            circum_map[name]=calculate_pixel_distance(coordinate[0],coordinate[1])
    return circum_map
