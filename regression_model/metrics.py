import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

def calculate_regression_metrics(ypred, ytrue,inference=False):
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(ypred - ytrue))

    # Mean Squared Error (MSE)
    mse = np.mean(np.square(ypred - ytrue))

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)

    # R-squared (R2) Score
    ssr = np.sum(np.square(ypred - ytrue))
    sst = np.sum(np.square(ytrue - np.mean(ytrue)))
    r2 = 1 - (ssr / sst)

    # Adjusted R-squared (Adjusted R2) Score
    n = len(ytrue)
    p = len(ypred)
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'Adjusted R2': adjusted_r2
    }
    if inference:
        # Print all metrics
        for metric, value in metrics.items():
            print(f'{metric}: {value}')

    return metrics

# Example usage
ypred = np.array([2.5, 3.7, 4.1, 5.0])
ytrue = np.array([2.0, 4.0, 3.5, 4.8])
calculate_regression_metrics(ypred, ytrue)
