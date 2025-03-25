import pandas as pd
import os
import glob
from main import load_data, preprocess_data, handle_device_specific_processing, group_and_resample

def remove_columns_by_name(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    print("[5/9] Removing specified columns...")
    # Remove only the columns that exist in the DataFrame
    columns_to_remove = [col for col in columns if col in df.columns]
    df = df.drop(columns=columns_to_remove)
    print("Columns removed:\n", df.head())
    return df

def main(file_path: str):
    print(f"Starting main process for file: {file_path}...")

    df = load_data(file_path)
    df = preprocess_data(df)
    df = handle_device_specific_processing(df)

    # Group and resample data
    grouped_data = group_and_resample(df, group_cols=["Plot", "Depth level", "Horizontal distance"], resample_interval='10min')

    # Remove specific columns by name
    columns_to_remove = ["G2_60_100","G2_110_100", "G2_130_100"]  # Example columns to remove
    df = remove_columns_by_name(grouped_data, columns_to_remove)

    # Create the 'Import' directory if it doesn't exist
    import_directory = os.path.join(os.getcwd(), "Import")
    os.makedirs(import_directory, exist_ok=True)

    # Define the output file path with the original name and "processed" suffix
    original_file_name = os.path.basename(file_path)
    name_without_extension = os.path.splitext(original_file_name)[0]
    output_file_name = f"{name_without_extension}_preprocessed.csv"
    output_file_path = os.path.join(import_directory, output_file_name)

    # Save the processed data as a CSV file in the 'Import' directory
    df.to_csv(output_file_path)

    print(f"Main process completed for file: {file_path}. Data saved to {output_file_path}.")

if __name__ == "__main__":
    data_directory = os.path.join(os.getcwd(), "data")
    file_pattern = os.path.join(data_directory, "data_cs655.csv")

    # Process only the first file found in the directory
    file_paths = glob.glob(file_pattern)
    if file_paths:
        main(file_paths[0])
    else:
        print("No files found in the directory.")