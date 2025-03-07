import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os
from pathlib import Path


def detect_outliers(
        df: pd.DataFrame,
        window_size: int = 24,
        max_std_factor: float = 3.0,
        min_window_values: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect outliers in time series data using a trailing window approach.
    Outliers are excluded from subsequent window calculations.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing time series data with timestamps as index
    window_size : int, default=24
        Size of the trailing window for calculating statistics
    max_std_factor : float, default=3.0
        Maximum number of standard deviations before a value is considered an outlier
    min_window_values : int, default=5
        Minimum number of non-NaN values required in the window for outlier detection

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        First DataFrame contains the cleaned time series (without outliers)
        Second DataFrame contains only the outliers
    """
    # Initialize result DataFrames
    clean_df = df.copy()
    outliers_df = pd.DataFrame(index=df.index, columns=df.columns, dtype=df.dtypes.iloc[0])
    # Working DataFrame - will have outliers marked as NaN as we go
    working_df = df.copy()

    # Process each column separately
    for column in df.columns:
        print(f"Processing outliers for column: {column}")
        # Skip the first window_size values (not enough history for window calculation)
        for i in range(window_size, len(df)):
            # Extract the trailing window (excluding already identified outliers)
            window = working_df[column].iloc[i - window_size:i].dropna()

            # Check if we have enough values in the window
            if len(window) >= min_window_values:
                # Calculate statistics
                window_mean = window.mean()
                window_std = window.std()

                # Set threshold boundaries
                upper_bound = window_mean + (max_std_factor * window_std)
                lower_bound = window_mean - (max_std_factor * window_std)

                # Current value
                current_value = df[column].iloc[i]

                # Check if current value is an outlier (and not NaN)
                if pd.notna(current_value) and (current_value > upper_bound or current_value < lower_bound):
                    # Mark as outlier
                    outliers_df.iloc[i, outliers_df.columns.get_loc(column)] = current_value
                    clean_df.iloc[i, clean_df.columns.get_loc(column)] = np.nan
                    working_df.iloc[i, working_df.columns.get_loc(column)] = np.nan

    return clean_df, outliers_df


def count_gaps(
        df: pd.DataFrame,
        bucket_interval: str = '1D',
        resample_interval: str = None
) -> pd.DataFrame:
    """
    Count gaps in time series data and group them into buckets.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing time series data with timestamps as index
    bucket_interval : str, default='1D'
        Pandas-compatible interval string for bucketizing gaps (e.g., '1D', '12H', '30min')
    resample_interval : str, default=None
        If specified, resample the data to this interval before counting gaps.
        If None, use the existing time index.

    Returns:
    --------
    pd.DataFrame
        DataFrame with time series names as rows, gap duration buckets as columns,
        and the count of gaps in each bucket as values
    """
    # Resample data if specified
    if resample_interval is not None:
        print(f"Resampling data to {resample_interval} interval...")
        df_resampled = df.resample(resample_interval).mean()
    else:
        df_resampled = df.copy()

    # Convert bucket interval to timedelta
    bucket_td = pd.to_timedelta(bucket_interval)

    # Default number of buckets - updated to include 0-1 interval
    max_gap_buckets = 11

    # Create bucket labels - now starting from 0
    bucket_labels = []
    for i in range(max_gap_buckets):
        bucket_start = i
        bucket_end = i + 1
        bucket_labels.append(f"{bucket_start}-{bucket_end} {bucket_interval}")

    # Initialize results dictionary
    results_dict = {}

    # Process each column
    for column in df_resampled.columns:
        print(f"Counting gaps for column: {column}")
        gap_counts = [0] * max_gap_buckets

        # Get NaN mask
        is_nan = df_resampled[column].isna()

        if is_nan.any():
            # Find consecutive NaN values
            consecutive_nans = []
            current_gap = []

            for idx, value in enumerate(is_nan):
                if value:  # If True (is NaN)
                    current_gap.append(idx)
                elif current_gap:  # If not NaN but we have a gap accumulating
                    consecutive_nans.append(current_gap)
                    current_gap = []

            # Handle case where series ends with NaNs
            if current_gap:
                consecutive_nans.append(current_gap)

            # Calculate gap durations and assign to buckets
            for gap in consecutive_nans:
                if not gap:  # Skip empty gaps
                    continue

                # Calculate gap duration in terms of index units
                if isinstance(df_resampled.index, pd.DatetimeIndex):
                    first_idx = gap[0]
                    last_idx = gap[-1]
                    if first_idx < len(df_resampled.index) and last_idx < len(df_resampled.index):
                        gap_start = df_resampled.index[first_idx]
                        gap_end = df_resampled.index[last_idx]
                        gap_duration = gap_end - gap_start

                        # Convert to bucket units - include 0 as starting index
                        bucket_idx = min(int(gap_duration / bucket_td), max_gap_buckets - 1)
                        gap_counts[bucket_idx] += 1
                else:
                    # For non-datetime index, just use the length
                    gap_length = len(gap)
                    # Now includes 0-based indexing
                    bucket_idx = min(gap_length, max_gap_buckets - 1)
                    gap_counts[bucket_idx] += 1

        results_dict[column] = gap_counts

    # Convert to DataFrame
    results_df = pd.DataFrame(results_dict).T
    results_df.columns = bucket_labels

    return results_df


def plot_outliers(df: pd.DataFrame, clean_df: pd.DataFrame, outliers_df: pd.DataFrame,
                  columns: List[str] = None, output_dir: str = "plots") -> None:
    """
    Plot time series data with outliers highlighted.

    Parameters:
    -----------
    df : pd.DataFrame
        Original data
    clean_df : pd.DataFrame
        Clean data (with outliers removed)
    outliers_df : pd.DataFrame
        DataFrame containing only outliers
    columns : List[str], default=None
        List of columns to plot. If None, plot all columns.
    output_dir : str, default="plots"
        Directory to save plots
    """
    if columns is None:
        columns = df.columns

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for column in columns:
        plt.figure(figsize=(12, 6))

        # Plot original data
        plt.plot(df.index, df[column], 'b-', alpha=0.3, label='Original')

        # Plot clean data
        plt.plot(df.index, clean_df[column], 'g-', label='Clean')

        # Plot outliers
        outlier_mask = outliers_df[column].notna()
        if outlier_mask.any():
            plt.scatter(
                outliers_df.index[outlier_mask],
                outliers_df[column][outlier_mask],
                color='r',
                marker='x',
                s=50,
                label='Outliers'
            )

        plt.title(f'Outlier Detection: {column}')
        plt.xlabel('Timestamp')
        plt.ylabel('Value')
        plt.legend()
        plt.tight_layout()

        # Save figure
        plt.savefig(os.path.join(output_dir, f'outliers_{column}.png'))
        plt.close()


def process_file(file_path: str, output_prefix: str) -> None:
    """
    Process a single file for outlier detection and gap counting.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    output_prefix : str
        Prefix for output files and directory names
    """
    print(f"\nProcessing file: {file_path}")

    # Create output directories
    export_dir = os.path.join('Export', output_prefix)
    outliers_dir = os.path.join(export_dir, 'outliers')
    plots_dir = os.path.join(export_dir, 'plots')

    os.makedirs(export_dir, exist_ok=True)
    os.makedirs(outliers_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Load the CSV file directly
    df = pd.read_csv(file_path)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Simple preprocessing: convert Timestamp to datetime and set as index
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce')

    # Get the Value column and set it to numeric
    df["Value"] = pd.to_numeric(df["Value"], errors='coerce')

    # Get the time series data for analysis
    # We need to organize by Plot and Depth level
    time_series_df = df.pivot_table(
        index="Timestamp",
        columns=["Plot", "Depth level"],
        values="Value",
        aggfunc='first'  # Just take the first value if there are duplicates
    )

    # Flatten the MultiIndex columns for easier handling
    time_series_df.columns = [f"{col[0]}_{col[1]}" for col in time_series_df.columns]

    if time_series_df.empty:
        print("No data to process after extraction.")
        return

    print(f"Extracted {len(time_series_df.columns)} time series.")

    # Outlier detection
    print("\nRunning outlier detection...")
    window_size = 5  # Smaller window for the sample data
    max_std_factor = 2.0
    min_window_values = 3

    clean_df, outliers_df = detect_outliers(
        time_series_df,
        window_size=window_size,
        max_std_factor=max_std_factor,
        min_window_values=min_window_values
    )

    print(f"Clean data has {clean_df.count().sum()} values")
    print(f"Detected {outliers_df.count().sum()} outliers")

    # Save outlier detection results
    clean_df.to_csv(os.path.join(outliers_dir, 'clean_data.csv'))
    outliers_df.to_csv(os.path.join(outliers_dir, 'outliers_data.csv'))
    print(f"Outlier detection results saved to {outliers_dir}/ directory")

    # Plot outliers
    plot_outliers(time_series_df, clean_df, outliers_df, output_dir=plots_dir)

    # Gap counting
    print("\nRunning gap counting...")
    bucket_interval = '1D'
    resample_interval = '10min'

    gap_counts = count_gaps(
        clean_df,  # Use clean data to count gaps
        bucket_interval=bucket_interval,
        resample_interval=resample_interval
    )

    print("\nGap counts by bucket:")
    print(gap_counts)

    # Save gap counts
    gap_counts.to_csv(os.path.join(export_dir, 'gap_counts.csv'))
    print(f"Gap counts saved to {export_dir}/gap_counts.csv")


def main() -> None:
    """
    Main function to test the outlier detection and gap counting functions.
    """
    # File paths
    agvo_file = 'data/data_agvo.csv'
    cs655_file = 'data/data_cs655.csv'

    # Make sure the Export directory exists
    os.makedirs('Export', exist_ok=True)

    # Process each file individually
    for file_path, prefix in [(agvo_file, 'agvo'), (cs655_file, 'cs655')]:
        if Path(file_path).exists():
            process_file(file_path, prefix)
        else:
            print(f"File {file_path} not found.")


if __name__ == "__main__":
    main()