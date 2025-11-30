# -*- coding: utf-8 -*-
"""
Flood damage validation.
"""

import os
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
        self._drop_missing()
        
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

    def _drop_missing(self):
        """
        Drop rows with missing prediction or observation values.

        Notes
        -----
        Rows with NaN in either prediction or observation column
        are removed to avoid issues in metric calculations.
        """
        before = len(self.df)
        self.df = self.df.dropna(subset=[self.prediction_column, self.observed_column])
        after = len(self.df)
        dropped = before - after

        if dropped > 0:
            print(
                f"[INFO] Dropped {dropped} rows with missing values in "
                f"'{self.prediction_column}' or '{self.observed_column}'."
            )

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
            and Bias (pred - obs, positive = overestimation)
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
    
    def compute_loss_metrics(self, value_col="replacement_value_kEUR"):
            """
            Compute performance metrics for financial losses

            Parameters
            ----------
            value_col : str, optional
                Name of the column with asset replacement values (e.g. in kEUR),
                by default "replacement_value_kEUR".

            Returns
            -------
            dict
                Dictionary with the following keys:
                - "MAE_loss"
                - "RMSE_loss"
                - "Bias_loss"

            Raises
            ------
            ValueError
                If `value_col` is not present in the dataset.
            """
            df = self.dataset.df

            if value_col not in df.columns:
                raise ValueError(f"Column '{value_col}' not found in dataset.")

            dr_pred = self.dataset.df[self.dataset.prediction_column].to_numpy()
            dr_obs = self.dataset.df[self.dataset.observed_column].to_numpy()
            values = df[value_col].to_numpy()

            loss_pred = dr_pred * values
            loss_obs = dr_obs * values

            errors = loss_pred - loss_obs
            mae_loss = np.mean(np.abs(errors))
            rmse_loss = np.sqrt(np.mean(errors ** 2))
            bias_loss = np.mean(errors)

            return {
                "MAE_loss": mae_loss,
                "RMSE_loss": rmse_loss,
                "Bias_loss": bias_loss,
            }

    def compute_segment_means(self, segment_col):
        """
        Compute mean predicted and observed damage ratios per segment.
        
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
            If 'segment_col' is not present in the dataset.
        """
                
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
    def print_performance_summary(metrics):
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

    @staticmethod
    def print_loss_summary(loss_metrics):
        """
        Print a text summary of financial loss metrics.

        Parameters
        ----------
        loss_metrics : dict
            Dictionary as returned by `compute_loss_metrics`.
        """
        print("\nFinancial loss validation")
        print(f"MAE loss      : {loss_metrics['MAE_loss']:.3f}")
        print(f"RMSE loss     : {loss_metrics['RMSE_loss']:.3f}")
        print(f"Bias loss     : {loss_metrics['Bias_loss']:.3f}")

if __name__ == "__main__":
    # Example usage
    csv_path = os.path.join("data", "validation_data.csv")
    validation_data = ValidationDataset(csv_path)
    print(validation_data.df.head())
    
    validator = FloodDamageValidator(validation_data)
    print("\n", validator.compute_segment_means("building_type"))
    
    # Damage-ratio metrics
    metrics = validator.compute_metrics()
    FloodDamageValidator.print_performance_summary(metrics)
    
    # Financial loss metrics
    loss_metrics = validator.compute_loss_metrics("replacement_value_kEUR")
    validator.print_loss_summary(loss_metrics)
    
    # Scatter plot
    validator.plot_scatter()
