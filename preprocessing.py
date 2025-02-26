## To-Do
# für die Datenbereinigung

def main(file_path: str):
    print("Starting main process...")
    total_steps = 8

    df = load_data(file_path)
    df = preprocess_data(df)
    df = handle_device_specific_processing(df)

    grouped_data = group_and_resample(df, group_cols=["Plot", "Depth level", "Horizontal distance"], resample_interval='10min')

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
    file_path = os.path.join(os.getcwd(), "data", "data_agvo.csv")
    if Path(file_path).exists():
        main(file_path)
    else:
        print(f"File {file_path} not found.")