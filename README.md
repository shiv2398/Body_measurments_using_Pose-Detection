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

1. **Approximation of Measurements**:
2. The first step is to approximate body measurements using pose landmarks from the image data.
3. Main file of this project can be run into three modes ```data approximation```,```training```,```testing```
### Data approximation : 
 1. In this phase main file use several functions to extract the joint distances using the pose detection model and save it into a data frame
 2. command for data approximation

** Training** 
1. In this mode main function use more than 10 models of regressions to map the approximate values to target values
2. it also provide several functionality
3. single model functionality (approximate all the column with one regression model)
4. Multiple model functionality (approximate all the columns by different models on each columns )
   

**Testing**
1.Testing mode is only for user , it takes an input image and then return body measurements in inches and cm also.






