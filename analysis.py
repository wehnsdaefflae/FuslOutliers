import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import glob
from time_series_analysis import detect_outliers, count_gaps, plot_outliers
from main import load_data, get_experiments, group_experiments, process_experiments, plot_and_export_group_data
import pytz

def calculate_group_differences(df: pd.DataFrame) -> pd.DataFrame:
    print("Calculating group differences...")
    differences = {}
    for col1 in df.columns:
        for col2 in df.columns:
            if col1 != col2:
                diff_col_name = f"{col1}_minus_{col2}"
                differences[diff_col_name] = df[col1] - df[col2]
    return pd.DataFrame(differences)

def main(file_path: str):
    print(f"Starting main process for file: {file_name}...")
    total_steps = 8
    file_name_new = file_name.rsplit("_", 1)[0]
    export_base_path = os.path.join(os.getcwd(), "Export", file_name_new)
    os.makedirs(export_base_path, exist_ok=True)
    grouped_data = load_data(file_path)
    grouped_data["Timestamp"] = pd.to_datetime(grouped_data["Timestamp"], errors='coerce')
    grouped_data.set_index("Timestamp", inplace=True)

    window_size = 24
    max_std_factor = 3.0
    min_window_values = 6
    clean_df, outliers_df = detect_outliers(grouped_data, window_size, max_std_factor, min_window_values)
    bucket_interval = '1h'
    resample_interval = '1h'
    gap_counts = count_gaps(clean_df, bucket_interval, resample_interval)
    output_dir = os.path.join(export_base_path, "Sensoren")
    plot_outliers(grouped_data, clean_df, outliers_df, columns=grouped_data.columns, output_dir=output_dir)
    print(f"Plots saved to {output_dir}/ directory")
    print(f"Gap counts:\n{gap_counts}")

    df_experiments, groups = process_experiments(grouped_data)
    for group_name in df_experiments.columns:
        plot_and_export_group_data(grouped_data[groups[group_name]], df_experiments, group_name, file_name, export_base_path)

    df_experiments = df_experiments.resample('D').mean()

    def convert_index_to_column_and_format(df):
        # Überprüfen, ob der Index ein DatetimeIndex ist
        if isinstance(df.index, pd.DatetimeIndex):
            # Umwandeln in deutsche Zeit (CET/CEST)
            berlin_tz = pytz.timezone('Europe/Berlin')
            df.index = df.index.tz_convert(berlin_tz).tz_localize(None)
            # Index formatieren
            df.index = df.index.strftime('%d.%m.%Y %H:%M:%S')

            # Index zurücksetzen, um ihn als Spalte zu behalten
            df = df.reset_index()

        return df

    print(clean_df)
    print(outliers_df)
    print(df_experiments)

    # Zeitinformationen aus den DataFrames entfernen
    clean_df = convert_index_to_column_and_format(clean_df)
    outliers_df = convert_index_to_column_and_format(outliers_df)
    df_experiments = convert_index_to_column_and_format(df_experiments)



    export_path = f"{export_base_path}/{file_name_new}_processed.xlsx"
    with pd.ExcelWriter(export_path, engine='xlsxwriter') as writer:
        clean_df.to_excel(writer, sheet_name='Sensoren', index=False)
        outliers_df.to_excel(writer, sheet_name='Outliers', index=False)
        gap_counts.to_excel(writer, sheet_name='Lücken', index=True)
        df_experiments.to_excel(writer, sheet_name='Experimente', index=False)

    print(f"DataFrames wurden erfolgreich in {export_path} gespeichert.")

    print(f"Main process completed for file: {export_path}. All processing steps completed.")


if __name__ == "__main__":
    data_directory = os.path.join(os.getcwd(), "Import")
    file_pattern = os.path.join(data_directory, "*.csv")
    for file_path in glob.glob(file_pattern):
        file_name = os.path.basename(file_path)
        if Path(file_path).exists():
            main(file_path)
        else:
            print(f"File {file_path} not found.")