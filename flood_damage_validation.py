# -*- coding: utf-8 -*-
"""
Flood damage model validation tools.

This module provides an object-oriented framework for validating
predicted damage ratios against observed data.

Includes:
- loading validation data from CSV,
- computing global performance metrics (MAE, RMSE, Bias),
- evaluating a mean-based baseline model,
- segment-level analysis,
- basic visualization of model fit.
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
        
        # basic data checks
        self.check_model_damage_range()
    
    def check_model_damage_range(self, id_col="property_id"):
        """
        Check that damage-ratio values lie in [0.0, 1.0].

        - Only the 'DR_pred' column is checked.
        - Values outside [0.0, 1.0] or NaN are removed.
        - If any rows are removed, prints how many and lists the removed property_id's.
        - If nothing is removed, remains silent.
        - Updates self.dataset.df.

        Parameters
        ----------
        id_col : str, optional
            Column name identifying unique objects (default: 'property_id').

        Raises
        ------
        ValueError
            If 'DR_pred' or id_col are missing from the dataset.
        """

        df = self.dataset.df

        # required columns
        pred_col = self.dataset.prediction_column  # should be 'DR_pred'
        missing = [c for c in (pred_col, id_col) if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required column(s): {', '.join(missing)}")

        # valid mask: DR_pred not NaN and in [0, 1]
        valid_mask = df[pred_col].notna() & (df[pred_col] >= 0.0) & (df[pred_col] <= 1.0)

        # offending rows = NOT valid
        offending_mask = ~valid_mask
        n_offending = int(offending_mask.sum())

        if n_offending > 0:
            removed_ids = df.loc[offending_mask, id_col].astype(str).tolist()

            # remove invalid rows
            self.dataset.df = df.loc[valid_mask].reset_index(drop=True)

            print(
                f"[WARNING] Removed {n_offending} invalid row(s) in '{pred_col}' "
                f"(outside [0.0, 1.0] or NaN). Removed {id_col}: {removed_ids}"
            )

        # if none removed, remain silent
        return n_offending
    
    def check_monotonicity(self, depth_col="water_depth_m"):
        """
            Check whether observed damage ratios increase with flood depth.

            This method provides two checks:
            
            1. Correlation between flood depth and observed damage ratio.
            A positive value indicates that deeper flooding corresponds to 
            higher damage ratios.
            
            2. Predefined depth-bin stratification:
                Bins:
                    - 0–0.2 m
                    - 0.2–0.5 m
                    - 0.5–1.0 m
                    - 1.0–1.5 m
                    - > 1.5 m

                The mean observed damage ratio is computed for each bin.

            Parameters
            ----------
            depth_col : str, optional
                Name of the column representing flood depth. 
                Default is "water_depth_m".
            
            Returns
            -------
            dict
                Dictionary containing:
                - "correlation" : float
                    Pearson correlation coefficient depth to damage ratio.
                - "depth_bin_means" : pandas.Series
                    Mean observed damage ratio for each bin.

            Raises
            ------
            ValueError
                If 'depth_col' is missing from the dataset.
        """
        df = self.dataset.df

        if depth_col not in df.columns:
            raise ValueError(f"Column '{depth_col}' not found in dataset.")
        
        # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html
        corr = df[depth_col].corr(df[self.dataset.observed_column])
        
        bins = [0.0, 0.2, 0.5, 1.0, 1.5, float("inf")]
        labels = ["0-0.2", "0.2-0.5", "0.5-1.0", "1.0-1.5", ">1.5"]

        # https://pandas.pydata.org/docs/reference/api/pandas.cut.html
        df["_depth_bin"] = pd.cut(  # Convert continuous depths into predefined bins
            df[depth_col],
            bins=bins,
            labels=labels,
            include_lowest=True
        )

        depth_bin_means = (
            df.groupby("_depth_bin", observed=True)[self.dataset.observed_column]
            .mean()
        )

        # remove temp column
        df.drop(columns=["_depth_bin"], inplace=True)

        return {
            "correlation": corr,
            "depth_bin_means": depth_bin_means.round(3)
        }
    
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
        df = self.dataset.df
        
        dr_pred = df[self.dataset.prediction_column].to_numpy()
        dr_obs = df[self.dataset.observed_column].to_numpy()
        
        errors = dr_pred - dr_obs
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean((errors) ** 2))
        bias = np.mean(errors)
        
        return {
            "MAE": mae,
            "RMSE": rmse,
            "Bias": bias,
        }
    
    def compute_segment_metrics(self, segment_col):
        """
        Compute MAE, RMSE and Bias per segment (category).
        
        Parameters
        ----------
        segment_col : str
            Name of the column used for grouping.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame containing error metrics for each segment with columns:
            - "MAE"
            - "RMSE"
            - "Bias"

        Raises
        ------
        ValueError
            If 'segment_col' is not present in the dataset.
        """
                
        df = self.dataset.df
        
        if segment_col not in df.columns:
            raise ValueError(f"Column '{segment_col}' not found in dataset.")
        
        df_pred = df[self.dataset.prediction_column]
        df_obs = df[self.dataset.observed_column]
        
        df["_error"] = df_pred - df_obs
        df["_abs_error"] = np.abs(df["_error"])
        df["_squared_error"] = df["_error"] ** 2
        
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html
        # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.agg.html
        grouped = (
            df.groupby(segment_col)
                .agg(
                    N = ("_error", "size"),
                    MAE = ("_abs_error", "mean"),
                    RMSE = ("_squared_error", lambda x: np.sqrt(np.mean(x))),
                    Bias = ("_error", "mean"),
                )
                .reset_index()
        )
        
        # remove temp cols
        df.drop(columns=["_error", "_abs_error", "_squared_error"], inplace=True)
        
        return grouped.round(3)
   
    def compute_baseline_metrics(self):
        """
        Compute baseline metrics using mean observed damage ratio as prediction.

        Returns
        -------
        dict
            Dictionary containing MAE, RMSE, and Bias for the baseline model. Keys are:
            - "MAE_baseline"
            - "RMSE_baseline"
            - "Bias_baseline"
        """
        df = self.dataset.df
        
        dr_obs = df[self.dataset.observed_column].to_numpy()
        dr_pred_baseline = np.full_like(dr_obs, np.mean(dr_obs))  # new array with constant values equal to mean observed DR
        
        errors = dr_pred_baseline - dr_obs
        mae_baseline = np.mean(np.abs(errors))
        rmse_baseline = np.sqrt(np.mean(errors ** 2))
        bias_baseline = np.mean(errors)
        
        return {
            "MAE_baseline": mae_baseline,
            "RMSE_baseline": rmse_baseline,
            "Bias_baseline": bias_baseline,
        }
   
    def plot_depth_monotonicity(self, depth_col="water_depth_m"):
        """
        Plot mean observed damage ratio across predefined flood-depth bins.

        Uses depth-bin statistics computed by 'check_monotonicity' and
        visualises mean observed damage ratios per depth interval.

        Parameters
        ----------
        depth_col : str, optional
            Name of the column containing flood depths. Default is "water_depth_m".

        Raises
        ------
        ValueError
            If the depth column does not exist in the dataset.
        """

        # Reuse statistics from check_monotonicity (no need to repeat binning logic)
        stats = self.check_monotonicity(depth_col=depth_col)
        depth_means = (
            stats["depth_bin_means"]
            .reset_index()
            .rename(columns={self.dataset.observed_column: "mean_damage"})
        )

        plt.figure(figsize=(8, 5))
        sns.set_style("whitegrid")

        sns.barplot(
            data=depth_means,
            x=depth_means.columns[0],   # depth-bin category
            y="mean_damage",
            hue=depth_means.columns[0],          # required to avoid warning
            palette="Blues",
            legend=False
        )

        plt.xlabel("Flood depth interval [m]")
        plt.ylabel("Mean observed damage ratio")
        plt.title("Observed Damage Ratio Across Flood-Depth Bins")

        plt.ylim(0, max(depth_means["mean_damage"]) * 1.15)
        plt.tight_layout()
        plt.show()
   
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
        
        # determine max value and round it up to the next 0.1
        raw_max = max(dr_obs.max() + 0.05, dr_pred.max() + 0.05)
        range_max = np.ceil(raw_max * 10) / 10.0
        
        # diagonal reference line starting at (0,0)
        plt.plot([0, range_max], [0, range_max],
                linestyle="--", color="red", label="1:1 line")

        # seaborn scatter
        sns.scatterplot(
            data=df,
            x=self.dataset.observed_column,
            y=self.dataset.prediction_column,
            hue="building_type",
            palette="Set2",
            s=70
        )
        
        # enforce axes starting at 0 and having the same range
        plt.xlim(0, range_max)
        plt.ylim(0, range_max)

        plt.xlabel("Observed Damage Ratio")
        plt.ylabel("Predicted Damage Ratio")
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
        
        if metrics['MAE'] < 0.05:
            quality = "very good"
        elif metrics['MAE'] < 0.10:
            quality = "moderate"
        else:
            quality = "poor"
        
        print(f"Model fit (based on MAE): {quality}")

    @staticmethod
    def print_baseline_summary(baseline_metrics, model_metrics=None):
        """
        Print a summary of baseline performance metrics.

        Parameters
        ----------
        baseline_metrics : dict
            Metrics dictionary as returned by 'compute_baseline_metrics'.
        model_metrics : dict, optional
            Metrics for the main model (from 'compute_metrics'), used to
            show relative improvement in MAE if provided.
        """

        print("\nGlobal baseline metrics (constant Damage Ratio = mean observed)")
        print(f"Baseline MAE   : {baseline_metrics['MAE_baseline']:.3f}")
        print(f"Baseline RMSE  : {baseline_metrics['RMSE_baseline']:.3f}")
        print(f"Baseline Bias  : {baseline_metrics['Bias_baseline']:.3f}")

        if model_metrics is not None and "MAE" in model_metrics:
            mae_model = model_metrics["MAE"]
            if baseline_metrics['MAE_baseline'] > 0:
                improvement = (1.0 - mae_model / baseline_metrics['MAE_baseline']) * 100.0
                if improvement < 0:
                    print(f"Model performs worse than baseline by {abs(improvement):.1f}%.")
                else:
                    print(f"MAE improvement vs baseline: {improvement:.1f}%.")


if __name__ == "__main__":
    # Example usage
    csv_path = os.path.join("data", "validation_data.csv")
    validation_data = ValidationDataset(csv_path)
    print(validation_data.df.head())
    
    validator = FloodDamageValidator(validation_data)
    print("\n", validator.compute_segment_metrics("building_type"))
    
    # Damage-ratio metrics
    metrics = validator.compute_metrics()
    baseline_metrics = validator.compute_baseline_metrics()
    validator.print_baseline_summary(baseline_metrics, model_metrics=metrics)
    validator.print_performance_summary(metrics)
    
    # Scatter plot
    validator.plot_depth_monotonicity()
    validator.plot_scatter()
