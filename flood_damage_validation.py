# -*- coding: utf-8 -*-
"""
Flood damage validation.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class ValidationDataset:
    """Keeps validation data for flood damage assessment."""
    
    def __init__(self, csv_path, prediction_column="DR_pred", observed_column="DR_obs"):
        """
        DR - Damage Ratio
        
        Parameters
        ----------
        csv_path : str
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

    def compute_segment_means(self, segment_col):
        """Compute mean predicted and observed damage ratios per segment.

            Parameters
            ----------
            segment_col : str
                Name of the column used for grouping.

            Returns
            -------
            pandas.DataFrame
                Mean DR_pred and DR_obs for each value of 'segment_col'.

            Raises
            ------
            ValueError
                If 'segment_col' is not present in the dataset."""
                
        df = self.dataset.df
        
        if segment_col not in df.columns:
            raise ValueError(f"Column '{segment_col}' not found in dataset.")
        
        grouped = df.groupby(segment_col)[[self.dataset.prediction_column, self.dataset.observed_column]].mean()
        grouped.columns = ["DR_pred_mean", "DR_obs_mean"]
        
        return grouped.round(3)

    def plot_scatter(self):
        """
        Create a scatter plot comparing predicted vs observed damage ratios
        using seaborn for better visual styling.
        """
        df = self.dataset.df
        dr_pred = df[self.dataset.prediction_column]
        dr_obs = df[self.dataset.observed_column]

        plt.figure(figsize=(6, 6))
        sns.set_style("whitegrid")

        # seaborn scatter
        sns.scatterplot(x=dr_obs, y=dr_pred, color="steelblue", s=70)
        
        # determine max value and round it up to the next 0.1
        raw_max = max(dr_obs.max() + 0.05, dr_pred.max() + 0.05)
        range_max = np.ceil(raw_max * 10) / 10.0

        # diagonal reference line starting at (0,0)
        plt.plot([0, range_max], [0, range_max],
                linestyle="--", color="red", label="1:1 line")

        # enforce axes starting at 0 and having the same range
        plt.xlim(0, range_max)
        plt.ylim(0, range_max)

        plt.xlabel("Observed DR")
        plt.ylabel("Predicted DR")
        plt.title("Predicted vs Observed Damage Ratios")
        plt.legend()

        plt.tight_layout()
        plt.show()


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
        print(f"Bias (pred-obs): {metrics['Bias']:.3f}\n")

if __name__ == "__main__":
    # Example usage
    validation_data = ValidationDataset(r"data\validation_data.csv")
    print(validation_data.df.head())
    validator = FloodDamageValidator(validation_data)
    metrics = validator.compute_metrics()
    FloodDamageValidator.print_summary(metrics)
    print(validator.compute_segment_means("building_type"))
    validator.plot_scatter()
