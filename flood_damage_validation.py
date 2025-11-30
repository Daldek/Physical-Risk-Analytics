# -*- coding: utf-8 -*-
"""
Flood damage validation.
"""

import numpy as np
import pandas as pd

class ValidationDataset:
    """Keeps validation data for flood damage assessment."""
    
    def __init__(self, csv_path, prediction_column="DR_pred", observed_column="DR_obs"):
        """
        DR - Damage Ratio
        
        Parameters
        ----------
        filename : str
            Path to an input csv file.
        prediction_column : str
            Name of the column with model predictions.
        observed_column : str
            Name of the column with observed values.
        """
        
        self.csv_path = csv_path
        self.prediction_column = prediction_column
        self.observed_column = observed_column
        
        # Load data
        self.df = pd.read_csv(self.csv_path)
        
        # df validation
        self._check_dataframe()
        
    def _check_dataframe(self):
        """
        Check if required columns are present in the dataframe.
        
        Raises
        ------
        ValueError
            If any of the required columns is missing in the CSV.
        """
        
        missing_columns = []
        if self.prediction_column not in self.df.columns:
            missing_columns.append(self.prediction_column)
        if self.observed_column not in self.df.columns:
            missing_columns.append(self.observed_column)
            
        if missing_columns:
            raise ValueError(f"Missing columns in the dataframe: {', '.join(missing_columns)}")


class FloodDamageValidator:
    """
    Validator for flood damage ratio predictions.

    This class computes global performance metrics such as MAE,
    RMSE and Bias between predicted and observed damage ratios.

    Parameters
    ----------
    dataset : ValidationDataset
        Validation dataset with predictions and observations.
    """
    
    def __init__(self, dataset):
        self.dataset = dataset
    
    def compute_metrics(self):
        """
        Compute global performance metrics.

        Returns
        -------
        dict
            Dictionary containing MAE (mean absolute error),
            RMSE (root mean squared error),
            and Bias (pred - obs)
        """
        
        dr_pred = self.dataset.df[self.dataset.prediction_column].values
        dr_obs = self.dataset.df[self.dataset.observed_column].values
        
        errors = dr_pred - dr_obs
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean((errors) ** 2))
        bias = np.mean(errors)
        
        return {
            "MAE": mae,
            "RMSE": rmse,
            "Bias": bias,
        }

    @staticmethod
    def print_summary(metrics):
        """
        Print a summary of performance metrics.

        Parameters
        ----------
        metrics : dict
            Metrics dictionary as returned by 'compute_metrics'.
        """
        print("\nGlobal validation metrics:")
        print(f"MAE            : {metrics['MAE']:.3f}")
        print(f"RMSE           : {metrics['RMSE']:.3f}")
        print(f"Bias (pred-obs): {metrics['Bias']:.3f}")

if __name__ == "__main__":
    # Example usage
    validation_data = ValidationDataset(r"data\validation_data.csv")
    print(validation_data.df.head())
    validator = FloodDamageValidator(validation_data)
    metrics = validator.compute_metrics()
    FloodDamageValidator.print_summary(metrics)
