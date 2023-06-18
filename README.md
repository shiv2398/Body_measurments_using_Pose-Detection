# Human Body Measurements using Deep Learning

This project aims to approximate body measurements using pose landmarks extracted from image data. It provides functionalities for data approximation, training regression models, and testing the accuracy of the approximated measurements.

## Installation
- Python 3.x
- TensorFlow (version latest)
- OpenCV (version latest)
- Pandas (version latest)
- scikit-learn (version latest)
- <add any other dependencies>
## Usage 
1. Install the required dependencies by running the following command:
 
 ``` pip install -r requirements.txt ```

## Data Approximation
The data approximation phase involves extracting joint distances using a pose detection model and saving the results into a data frame. To perform data approximation, follow these steps:



### Data Approximation Mode:

Command: ```python main.py data_approximation --csv_path <csv_path> --image_path <image_path>```

**Description:**

Performs data approximation using pose landmarks extracted from the image data.
Arguments:
`--csv_path`: Specifies the path to the CSV file containing the data.
`--image_path`: Specifies the path to the image file.


## Training

Regression Training Mode:

Command: ```python main.py regression_training --measurements_csv_path <measurements_csv_path> --separate_model_train <True/False> --model_name <model_name>```

**Description:**

 Performs regression model training to map the approximate values to target values.
 Arguments:
 `--measurements_csv_path:` Specifies the path to the CSV file containing the measurements data.
 `--separate_model_train:` Specifies whether to train one model for different columns (True/False).
 `--model_name:` Specifies the name of the model.


## Testing

The testing mode allows users to input an image and obtain body measurements in inches and centimeters. To perform testing, use the following command:


Command: ```python main.py testing --image_path <image_path>```

**Description:**

Performs testing with new image data to obtain body measurements.

 Arguments:
 `--image_path:` Specifies the path to the image file.









