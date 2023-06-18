import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from regression_model import regression_

class training:
    def __init__(self, single_model_inference=False, separate_model_inference=False):
        """
        Initializes the training class object.

        Args:
            single_model_inference (bool): Flag indicating single model inference.
            separate_model_inference (bool): Flag indicating separate model inference.
        """
        self.single_model_inference = single_model_inference
        self.separate_model_inference = separate_model_inference
        self.md_obj = regression_.models()

    def single_model(self, model_name, x, y):
        """
        Trains a single regression model.

        Args:
            model_name (str): Name of the regression model to be trained.
            x: Input features for training.
            y: Target variable for training.

        Returns:
            Trained regression model.
        """
        model, hyperparameters = self.md_obj.regression_models(model_name)
        print(f"Model: {model_name}, Hyperparameters: {hyperparameters}")
        model.fit(x, y)
        return model

    def ensemble_model(self, model_1, model_2, x_model_1, y_model_1, x_model_2, y_model_2):
        """
        Trains an ensemble of two regression models.

        Args:
            model_1 (str): Name of the first regression model.
            model_2 (str): Name of the second regression model.
            x_model_1: Input features for the first model.
            y_model_1: Target variable for the first model.
            x_model_2: Input features for the second model.
            y_model_2: Target variable for the second model.

        Returns:
            Trained models of the ensemble.
        """
        model_1, _ = self.md_obj.regression_models(model_1)
        model_2, _ = self.md_obj.regression_models(model_2)
        model_1.fit(x_model_1, y_model_1)
        model_2.fit(x_model_2, y_model_2)
        return model_1, model_2

    def neural_net_model_training(self, x, y):
        """
        Placeholder method for training neural network models.

        Args:
            x: Input features for training.
            y: Target variable for training.

        Returns:
            None
        """
        pass
