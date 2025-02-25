import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def load_data(file_path: str) -> pd.DataFrame:
    print("[1/8] Loading data...")
    return pd.read_csv(file_path)

## Was hältst du von einer data optimization?
## Würde hier reinpassen

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/8] Preprocessing data...")
    columns_to_keep = ["Timestamp", "Value", "Device", "Depth level", "Plot", "Horizontal distance", "Signal", "Sensor", "Depth"] ## "Unit", braucht man eig nicht
    df = df[columns_to_keep].copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce')
    df.set_index("Timestamp", inplace=True)

    # Filter rows where Unit indicates temperature (°C)
    # df = df[df["Unit"].str.contains("°C", na=False)] ## Der Code soll für alle Parameter gelten, nicht nur Temperatur
    df["Value"] = pd.to_numeric(df["Value"], errors='coerce')
    df.dropna(subset=["Value"], inplace=True)
    return df

def handle_device_specific_processing(df: pd.DataFrame) -> pd.DataFrame:
    print("[3/8] Handling device-specific processing...")

    # if "CS655" in df["Device"].unique(): ## CS655 muss nicht überarbeitet werden
    # df.loc[df["Device"] == "CS655", "Depth level"] = df.loc[df["Device"] == "CS655", "Depth level"].fillna(110)
    if "ENV__SOIL__T" in df["Signal"].unique():
        df.loc[df["Signal"] == "ENV__SOIL__T", "Device"] = "CLIMAVI"
        df.loc[df["Signal"] == "ENV__SOIL__T", "Depth level"] = df.loc[df["Signal"] == "ENV__SOIL__T", "Sensor"] * -1

    if "Pegel" in df["Device"].unique():
        df.loc[df["Device"] == "Pegel", "Depth level"] = df.loc[df["Device"] == "Pegel", "Depth"] + 60
        df.loc[df["Device"] == "Pegel", "Plot"] = df.loc[df["Device"] == "Pegel", "Plot"].str.replace("Pegelmessstelle",
                                                                                                      "Pegel")

    df.drop(columns=["Sensor", "Depth", "Device", "Signal"], inplace=True)

    return df

def group_and_resample(df: pd.DataFrame, group_cols: list, resample_interval: str = '10min') -> dict:
    print("[4/8] Grouping and resampling data...")
## Version aus altem Code überarbeitet
    grouped_data = df.groupby(group_cols)
    list_of_dfs = []
    for group_name, group_df in grouped_data:
        group_df = group_df.rename(columns={'Value': "_".join(map(str, group_name)).replace(" ", "_")})
        group_df = group_df.drop(columns=group_cols)
        list_of_dfs.append(group_df)
    data_sorted = list_of_dfs[0]
    for group_df in list_of_dfs[1:]:
        data_sorted = pd.merge(data_sorted, group_df, on='Timestamp', how='outer')
    grouped_data = data_sorted.resample(resample_interval, label='left').mean()
    return grouped_data

## Es werden einzelne Gruppen gebildet und damit weiter berechnet
    # grouped_data = {}
    # for group_name, group_df in df.groupby(group_cols):
    #     numeric_cols = group_df.select_dtypes(include=[np.number])
    #     group_resampled = numeric_cols.resample(resample_interval).mean()
    #     group_resampled = group_resampled.drop(columns=["Depth level", "Horizontal distance"], errors='ignore')
    #     group_name_str = "_".join(map(str, group_name)).replace(" ", "_")[:31]
    #     grouped_data[group_name_str] = group_resampled
    # return grouped_data

# def dynamic_mad_outliers(df: pd.DataFrame, window_size: int = 6, mad_multiplier: float = 20) -> pd.DataFrame:
    ## Ausreißer über Ableitung suchen

    # print("[5/8] Detecting outliers with dynamic MAD...")
    # result = df.copy()
    # for column in df.columns:
    #     medians = []
    #     mads = []
    #     for i in range(len(df)):
    #         start_idx = max(0, i - window_size + 1)
    #         end_idx = min(len(df), i + window_size)
    #         valid_values = df[column].iloc[start_idx:end_idx].dropna()
    #
    #         current_window_size = window_size
    #         while len(valid_values) < window_size and current_window_size <= len(df):
    #             current_window_size += 6
    #             start_idx = max(0, i - current_window_size + 1)
    #             end_idx = min(len(df), i + current_window_size)
    #             valid_values = df[column].iloc[start_idx:end_idx].dropna()
    #
    #         if len(valid_values) > 0:
    #             median = np.median(valid_values)
    #             mad = np.median(np.abs(valid_values - median))
    #             mad = max(mad, 1e-6)
    #         else:
    #             median = np.nan
    #             mad = np.nan
    #
    #         medians.append(median)
    #         mads.append(mad)
    #
    #     median_series = pd.Series(medians, index=df.index)
    #     mad_series = pd.Series(mads, index=df.index)
    #
    #     outlier_threshold = mad_series * mad_multiplier
    #     outliers = np.abs(df[column] - median_series) > outlier_threshold
    #     result[f'{column}_Outlier'] = outliers
    #
    # return result

## Eigentlich braucht man zu den einzelnen Sensoren keine min und max
# def calculate_group_statistics(grouped_data: dict) -> pd.DataFrame:
#     print("[6/8] Calculating group statistics...")
#     stats = {}
#     for group_name, group_df in grouped_data.items():
#         stats[group_name] = {
#             'mean': group_df.mean(),
#             'min': group_df.min(),
#             'max': group_df.max()
#         }
#     return stats

def export_grouped_data(export_path: str, grouped_data: dict):
    print("[7/8] Exporting grouped data...")
    os.makedirs(export_path, exist_ok=True)
    for name, df in grouped_data.items():
        csv_path = os.path.join(export_path, f"{name}.csv")
        df.to_csv(csv_path)

    excel_path = os.path.join(export_path, "grouped_data.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        for name, df in grouped_data.items():
            if not df.empty:
                # Ensure all datetime indices are timezone-unaware
                df = df.copy()
                if isinstance(df.index, pd.DatetimeIndex):
                    df.index = df.index.tz_localize(None)
                writer_sheet_name = name[:31]
                df.to_excel(writer, sheet_name=writer_sheet_name)
    print(f"Excel file exported to: {excel_path}")

# Additional Features

def format_group_label(group):
    if isinstance(group, tuple):
        return f"{group[0]} ({group[1]}, {group[2]})"
    return str(group)

def plot_data(df: pd.DataFrame, outliers_df: pd.DataFrame, export_path: str):
    print("[8/8] Plotting data...")
    os.makedirs(export_path, exist_ok=True)
    for column in df.columns:
        outliers = outliers_df.get(f'{column}_Outlier', pd.Series(dtype=bool))
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df[column], label='Data', color='blue')
        if not outliers.empty:
            plt.scatter(df.index[outliers], df[column][outliers], color='red', label='Outliers', s=20)
        plt.title(f"{column} Data with Outliers")
        plt.xlabel("Timestamp")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(export_path, f"{column}_plot.png"))
        plt.close()

def calculate_group_differences(df: pd.DataFrame) -> pd.DataFrame:
    print("Calculating group differences...")
    differences = {}
    for col1 in df.columns:
        for col2 in df.columns:
            if col1 != col2:
                diff_col_name = f"{col1}_minus_{col2}"
                differences[diff_col_name] = df[col1] - df[col2]
    return pd.DataFrame(differences)

def create_experiment_groups(df: pd.DataFrame) -> dict:
    print("Grouping by experiments...")
    groups = {}
    for column in df.columns:
        parts = column.split('_')
        if len(parts) > 1:
            experiment = parts[0]
            if experiment not in groups:
                groups[experiment] = []
            groups[experiment].append(column)
    return groups

def plot_experiment_outliers(df: pd.DataFrame, outliers_df: pd.DataFrame, experiment_groups: dict, export_path: str):
    print("Plotting experiment-level outliers...")
    os.makedirs(export_path, exist_ok=True)
    for experiment, columns in experiment_groups.items():
        if not columns:
            print(f"Skipping empty experiment group: {experiment}")
            continue
        plt.figure(figsize=(12, 6))
        for column in columns:
            plt.plot(df.index, df[column], label=f'Data ({column})', alpha=0.6)
            if f'{column}_Outlier' in outliers_df:
                outliers = outliers_df[f'{column}_Outlier']
                plt.scatter(df.index[outliers], df[column][outliers], label=f'Outliers ({column})', s=20)
        plt.title(f"Experiment: {experiment}")
        plt.xlabel("Timestamp")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(export_path, f"{experiment}_outliers.png"))
        plt.close()

def main(file_path: str):
    print("Starting main process...")
    total_steps = 8

    df = load_data(file_path)
    print(df)
    df = preprocess_data(df)
    print(df)
    df = handle_device_specific_processing(df)
    print(df)

    grouped_data = group_and_resample(df, group_cols=["Plot", "Depth level", "Horizontal distance"], resample_interval='10min')
    print(grouped_data)

    export_path = os.path.join(os.getcwd(), "Export")
    os.makedirs(export_path, exist_ok=True)

    all_outliers = {}
    experiment_groups = {}

    for group_name, group_df in grouped_data.items():
        outliers_df = dynamic_mad_outliers(group_df)
        all_outliers[group_name] = outliers_df

        group_differences = calculate_group_differences(group_df)

        plot_folder = os.path.join(export_path, "Plots", group_name)
        plot_data(group_df, outliers_df, plot_folder)

        experiment_groups[group_name] = create_experiment_groups(group_df)
        experiment_plot_folder = os.path.join(export_path, "Plots", "Experiments", group_name)
        plot_experiment_outliers(group_df, outliers_df, experiment_groups[group_name], experiment_plot_folder)

        group_df.to_csv(os.path.join(export_path, f"{group_name}_cleaned.csv"))
        group_differences.to_csv(os.path.join(export_path, f"{group_name}_differences.csv"))

    export_grouped_data(export_path, grouped_data)

    print("Main process completed. All processing steps completed.")

if __name__ == "__main__":
    file_path = os.path.join(os.getcwd(), "data", "test_data.csv")
    if Path(file_path).exists():
        main(file_path)
    else:
        print(f"File {file_path} not found.")
