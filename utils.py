import math
import csv
import pandas as pd 
import math
import yaml
import os 
import sys
from PIL import Image
from scipy.special import ellipe
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'configuration_files/configuration.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

DATA_CONFIG = load_config()



def pixel_to_cm(pixel_distance, ppi):
    cm_distance = (pixel_distance / ppi) * 2.54
    return cm_distance

def get_image_ppi(image_path):
    # Open the image using PIL
    image = Image.open(image_path)
    
    # Get the image resolution
    width, height = image.size
    
    # Get the physical size of the image (if available)
    try:
        dpi_width, dpi_height = image.info['dpi']
    except KeyError:
        # If DPI information is not available, assume 72 PPI
        dpi_width, dpi_height = 72, 72
    
    # Calculate the PPI for both width and height
    ppi_width = width / (dpi_width / 25.4)
    ppi_height = height / (dpi_height / 25.4)
    
    # Return the average PPI of the image
    return (ppi_width + ppi_height) / 2

def calculate_pixel_distance(joint1, joint2):
    # Calculate the Euclidean distance between two joints
    x1, y1 = joint1  # x, y coordinates of joint1
    x2, y2 = joint2  # x, y coordinates of joint2
    pixel_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return pixel_distance

def estimate_shoulder_width(pixels_shoulder_distance, average_shoulder_width):
    conversion_factor = average_shoulder_width / pixels_shoulder_distance
    return conversion_factor

def convert_to_cm(pixel_distance, conversion_factor):
    cm_distance = pixel_distance * conversion_factor
    return cm_distance

def circum_cal(cm_distance):
    a = (cm_distance) / 2  # Semi-major axis
    b = ((cm_distance) / 2) / 2  # Semi-minor axis
    
    # Check if a or b is zero
    if a == 0 or b == 0:
        return 0  # Return zero circumference if either a or b is zero
    
    # Calculate the value of the elliptic integral
    integral_value = ellipe((a**2 - b**2) / a**2)
    
    # Calculate the circumference
    circumference = 4 * a * integral_value
    return circumference
