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
        pred_col : str
            Name of the column with model predictions.
        obs_col : str
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
        """Check if required columns are present in the dataframe."""
        missing_columns = []
        if self.prediction_column not in self.df.columns:
            missing_columns.append(self.prediction_column)
        if self.observed_column not in self.df.columns:
            missing_columns.append(self.observed_column)
            
        if missing_columns:
            raise ValueError(f"Missing columns in the dataframe: {', '.join(missing_columns)}")

if __name__ == "__main__":
    # Example usage
    validation_data = ValidationDataset(r"data\validation_data.csv")
    print(validation_data.df.head())
