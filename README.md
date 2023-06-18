# Human Body Measurements using Deep Learning

This project aims to approximate human body measurements using deep learning techniques. It leverages pose landmarks extracted from image data to estimate various body measurements.

## Installation

To use this project, make sure you have the following dependencies installed:

- Python 3.x
- TensorFlow (version x.x.x)
- OpenCV (version x.x.x)
- Pandas (version x.x.x)
- scikit-learn (version x.x.x)
- <add any other dependencies>

You can install the required dependencies by running the following command:
```pip install -r requirements.txt```
## Usage

1. **Approximation of Measurements**: The first step is to approximate body measurements using pose landmarks from the image data. Use the following command to perform the approximation:

Replace `<image_directory>` with the path to the directory containing the image data and `<original_data_csv>` with the path to the original data CSV file.

2. **Training Regression Models**: After approximating the measurements, the next step is to train regression models to refine the approximated data and match it with the real values. Use the following command to train the regression models:


Replace `<original_data_csv>`, `<approximated_data_csv>`, `<columns_to_train>`, `<model_name>`, and `<model_save_path>` with the respective file paths and model configuration.

3. **Testing**: Once the regression models are trained, you can test them using new data. Use the following command to perform the testing:


Replace `<test_data_csv>` with the path to the test data CSV file and `<model_name>` with the name of the trained model.

Feel free to explore the code files for more detailed usage and customization options.




