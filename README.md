# Human Body Measurements using Deep Learning

This project aims to approximate body measurements using pose landmarks extracted from image data. It provides functionalities for data approximation, training regression models, and testing the accuracy of the approximated measurements.

## Installation
- Python 3.x
- TensorFlow (version latest)
- OpenCV (version latest)
- Pandas (version latest)
- scikit-learn (version latest)
- <add any other dependencies>

## Data Approximation
The data approximation phase involves extracting joint distances using a pose detection model and saving the results into a data frame. To perform data approximation, follow these steps:

1. Install the required dependencies by running the following command:
 
 ``` pip install -r requirements.txt ```

2. Run the main file in the data approximation mode using the following command:
`python3 Body_measurements/main.py data_approximation --csv_path Measurements.xlsx - Sheet1.csv --image_path Data`

## Training

The training mode focuses on mapping the approximate values to target values using various regression models. It offers the following functionalities:

- Single Model Functionality: Approximate all the columns with a single regression model. To use this functionality, run the following command:

- Multiple Model Functionality: Approximate each column with a different regression model. To use this functionality, run the following command:

## Testing

The testing mode allows users to input an image and obtain body measurements in inches and centimeters. To perform testing, use the following command:








