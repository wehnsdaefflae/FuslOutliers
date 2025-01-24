## Auswertung Datenportal mit Fehlersuche
## Hannes

## Import Tools
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
pd.options.mode.copy_on_write = True  # Avoid unnecessary deep copies
# pd.set_option('display.max_columns', 500) # Zeigt mehr Spalten an

## Import csv von Import Ordner
current_directory = os.getcwd()
file_path = os.path.join(current_directory, 'Import', 'März_warm_110.csv')
data = pd.read_csv(file_path, low_memory=False)
file_name = os.path.basename(file_path)
folder_name = os.path.splitext(file_name)[0] + " mit Fehlersuche"
print("Importierte csv:")
print(file_name)  # Kontrolle der importierten Datei
print()

## Optimize Data
def optimize_data_types(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif col_type == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
    return df
data = optimize_data_types(data)

## Vereinheitlichen der Sensoren
data_CS655 = data[data['Device'] == 'CS655'].copy()
data_Teros21 = data[data['Device'] == 'Teros21'].copy()
data_Pegel = data[data['Device'] == 'Pegel'].copy()
data_CLIMAVI = data[data['Signal'] == 'ENV__SOIL__T'].copy()
data_CLIMAVI['Device'] = 'CLIMAVI'
data_CLIMAVI['Depth level'] = data_CLIMAVI['Sensor'] * -1
data_CLIMAVI.drop(columns=['Sensor'], inplace=True)
data_Pegel['Depth level'] = data_Pegel['Depth'] + 60
data_Pegel['Plot'] = data_Pegel['Plot'].str.replace('Pegelmessstelle', 'Pegel')
data_device = pd.concat([data_CS655, data_Teros21, data_Pegel, data_CLIMAVI])  # Auswahl der Sensoren

## Löschen unnötiger Spalten
data_device = data_device.dropna(how='all')  # Leere Zeilen löschen
data_device = data_device.dropna(axis='columns')  # Leere Spalten löschen
print("Spalten:")
print(data_device.columns)  # Kontrolle der Spalten
print()
print('Anzahl leere Zellen:')
print(data_device.isnull().sum())  # Kontrolle der Zeilen
print()

## Umwandlung Datenformat
print('Datentyp:')
print(data_device.dtypes)
print()
data_device['Timestamp'] = pd.to_datetime(data_device.Timestamp, format='ISO8601', errors='coerce')  # Umwandlung in Zeitstempel
data_device['Depth level'] = data_device['Depth level'].astype('int16')  # Umwandlung der Tiefe
print('Kontrolle Datentyp:')
print(data_device.dtypes)
print()
print('Daten:')
print(data_device)

## Sortierung der Daten
data_grouped = data_device.groupby(['Plot', 'Depth level', 'Horizontal distance'])  # Gruppierung nach Spalten 'Plot', 'Depth level' und 'Horizontal distance'
list_of_dfs = []  # Erstellen neuer Dataframes für einzelnen Sensor
for group_name, group_df in data_grouped:
    group_df = group_df.drop(columns=['Plot', 'Depth level', 'Horizontal distance', 'Unit', 'Device', 'Depth', 'Signal'])  # Löschen unnötiger Spalten
    group_df = group_df.rename(columns={'Value': group_name})  # Neue Benennung
    list_of_dfs.append((group_name, group_df))
data_sorted = list_of_dfs[0][1]  # Erstellen eines Dataframes für alle Sensoren
for group_name, group_df in list_of_dfs[1:]:
    data_sorted = pd.merge(data_sorted, group_df, on='Timestamp', how='outer')

data_sorted.set_index('Timestamp', inplace=True)  # Einstellen der Zeit als Index
data_10min = data_sorted.resample('10min', label='left').mean()  # Vereinheitlichung der Zeit auf 10min Intervall (nur 1 Messwert)
print('Übersicht Sensoren:')
print(data_10min.columns)
print()
print('Daten pro Sensor:')
print(data_10min)

## Ausreißer anhand von rollendem Median Absolute Deviation mit dynamischem Window
def detect_outliers(df, window_size=6, required_values=6, mad_multiplier=20):
    result = df.copy()
    for column in df.columns:
        medians = []
        mads = []
        for i in range(len(df)):
            start_idx = max(0, i - window_size + 1)
            end_idx = min(len(df), i + window_size)
            valid_values = df[column].iloc[start_idx:end_idx].dropna()

            # Adjust window size if there are insufficient valid values
            current_window_size = window_size
            while len(valid_values) < required_values and current_window_size <= len(df):
                current_window_size += 6
                start_idx = max(0, i - current_window_size + 1)
                end_idx = min(len(df), i + current_window_size)
                valid_values = df[column].iloc[start_idx:end_idx].dropna()

            # Berechnung von Median und MAD
            if len(valid_values) > 0:
                median = np.median(valid_values)
                mad = np.median(np.abs(valid_values - median))

                while mad == 0 and len(valid_values) < len(df): # Falls MAD = 0, erweitere das Fenster weiter
                    current_window_size += 6  # Fenstergröße um 6 erhöhen
                    start_idx = max(0, i - current_window_size + 1)
                    end_idx = min(len(df), i + current_window_size)
                    valid_values = df[column].iloc[start_idx:end_idx].dropna()

                    # Rechne MAD erneut, falls gültige Werte gefunden werden
                    if len(valid_values) > 0:
                        mad = np.median(np.abs(valid_values - np.median(valid_values)))

                # Fallback für sehr kleine MAD-Werte
                mad = max(mad, 1e-6)
            else:
                median = np.nan
                mad = np.nan

            medians.append(median)
            mads.append(mad)

        # Create Median and MAD Series after processing all rows
        median_series = pd.Series(medians, index=df.index)
        mad_series = pd.Series(mads, index=df.index)

        # Calculate outliers
        outlier_threshold = mad_series * mad_multiplier
        outliers = np.abs(df[column] - median_series) > outlier_threshold

        # Add results to the DataFrame
        result[f'{column}_Outlier'] = outliers

    return result


# Apply the function to detect outliers
data_outliers = detect_outliers(data_10min)
# Create masks for unusual (outlier) values
unusual_mask = data_outliers.filter(like='Outlier').any(axis=1)

## Kontrolle Ausreißer
data_clean = data_10min.copy()  # Dataframe ohne Ausreißer
data_clean[unusual_mask] = np.nan
data_unusual = data_10min.copy()  # Dataframe für Ausreißer
data_unusual[~unusual_mask] = np.nan

# ## Ausreißer geplottet
# first_column = data_10min.columns[2]  # Select the first column
# outliers = data_outliers[f'{first_column}_Outlier']  # Get the outliers for the first column
# plt.figure(figsize=(10, 6))  # Plot the first column values
# # plt.plot(data_10min.index, data_10min[first_column], label='Data', color='blue')  # Liniendiagramm
# plt.scatter(data_10min.index, data_10min[first_column], label='Data', color='blue', s=10)  # Punktdiagramm
# plt.scatter(data_10min.index[outliers], data_10min[first_column][outliers], color='red', label='Outliers', s=20)  # Highlight the outliers
# # Adding labels and title
# plt.xlabel('Timestamp')
# plt.ylabel(f'{first_column} Values')
# plt.title(f'{first_column} with Outliers Highlighted')
# plt.legend()
# # Set the axis limits
# plt.xlim(pd.Timestamp('2023-03-1'), pd.Timestamp('2023-03-5'))
# plt.ylim(8.0, 9.5)
# plt.show()  # Show the plot

## Stündliche Daten
data_hourly = data_clean.resample('h', label='left').mean()  # left bedeutet 1 Uhr ->  1:00 bis 1:59
print('Stündliche Daten:')
print(data_hourly)

## Gruppierung nach Versuch
def get_group(column_name):
    parts = column_name.strip('()').split(', ')
    letter, number, last_num = parts[0][0], parts[1], parts[2]
    if last_num == '0':
        if letter in ['A', 'D', 'E', 'H']:
            return f'Ref {number}'
        elif letter in ['B', 'G']:
            return f'Warm {number}'
        elif letter in ['C', 'F']:
            return f'Kalt {number}'
        elif letter == 'X':
            return f'WarmX {number}'
        elif letter in ['P']:
            return f'Pegel {number}'
    elif last_num == '100':
        if letter == 'X':
            return f'DistX {number}'
        else:
            return f'Dist {number}'
    elif last_num == '300':
        if letter == 'X':
            return f'Dist3X {number}'
        else:
            return f'Dist3 {number}'
    return None

groups = {}
for column_tuple in data_hourly.columns:
    column = ", ".join(str(item) for item in column_tuple)
    group = get_group(column)
    if group:
        if group not in groups:
            groups[group] = []
        groups[group].append(column_tuple)  # Use tuple as key

data_Versuche = pd.DataFrame(index=data_hourly.index)
data_Versuche_min_max = pd.DataFrame(index=data_hourly.index)
data_unusual_2 = pd.DataFrame(index=data_hourly.index)

## Erstellen der Gruppen mit Fehlersuche
threshold_multiplier = 5

for group, columns in groups.items():
    group_data = data_hourly[columns]
    group_unusual = pd.DataFrame(index=data_hourly.index)

    for column in columns:
        # Calculate median and MAD over all columns in each row (axis=1)
        median = group_data.median(axis=1)  # Median over all columns in a row
        mad = np.median(np.abs(group_data.sub(median, axis=0)), axis=1)  # MAD over all columns in a row
        threshold = mad * threshold_multiplier

        # Identify outliers per row
        outliers = np.abs(group_data[column] - median) > threshold
        group_unusual[column] = np.where(outliers, group_data[column], np.nan)

        # Exclude outliers for mean calculation
        group_data[column] = np.where(outliers, np.nan, group_data[column])

    # Calculate mean of the group without outliers (axis=1 for row-wise mean)
    data_Versuche[group] = group_data.mean(axis=1)
    data_Versuche_min_max[f'{group}_min'] = group_data.min(axis=1)
    data_Versuche_min_max[f'{group}_max'] = group_data.max(axis=1)

    # Combine unusual data
    data_unusual_2 = pd.concat([data_unusual_2, group_unusual], axis=1)


# Select the first column group in data_Versuche
first_group = data_Versuche.columns[0]
# Extract corresponding original data and outliers
original_data_group = data_hourly[groups[first_group]]
outliers_group = data_unusual_2[groups[first_group]]

# # Plot all data
# plt.figure(figsize=(12, 6))
#
# # Plot each column of original data
# for column in original_data_group.columns:
#     plt.plot(data_hourly.index, original_data_group[column], label=f'Original Data ({column})', alpha=0.6)
#
# # Plot cleaned data (group mean remains as a single line)
# cleaned_data = data_Versuche[first_group]
# plt.plot(data_hourly.index, cleaned_data, label='Cleaned Data (Group Mean)', color='blue', linewidth=2)
#
# # Highlight outliers for each column
# for column in outliers_group.columns:
#     plt.scatter(data_hourly.index, outliers_group[column], label=f'Outliers ({column})', alpha=0.8, zorder=5)
#
# # Add labels, legend, and title
# plt.xlabel('Timestamp')
# plt.ylabel(f'{first_group} Values')
# plt.title(f'{first_group} Data with Outliers Highlighted')
# plt.legend(loc='upper left', fontsize='small', ncol=2)
# plt.grid(alpha=0.3)
# plt.tight_layout()
#
# # Set x and y axis limits
# # plt.xlim(pd.Timestamp('2023-03-05'), pd.Timestamp('2023-03-10'))
# # plt.ylim(5.5, 6.8)
#
# # Show the plot
# plt.show()

# Show the DataFrames
print("Versuche:")
print(data_Versuche)
print(data_Versuche_min_max)

# Define the desired order of groups
desired_order = ["WarmX", "DistX", "Dist3X", "Warm", "Dist", "Dist3", "Kalt", "Ref", "Pegel"]

# Define a function to extract and sort by the last characters of the column names
def custom_sort(column):
    parts = column.split()  # Split column name into parts
    key = parts[0]  # Get the first part (e.g., 'WarmX')
    suffix = parts[-1]  # Get the last part (e.g., '60', '110', '130')

    try:
        suffix_value = int(suffix)  # Attempt to convert suffix to integer
    except ValueError:
        suffix_value = float('inf')  # Use infinity if suffix is not an integer

    return (desired_order.index(key), suffix_value, column)

# Sort columns based on custom_sort function
sorted_columns = sorted(data_Versuche.columns, key=custom_sort)

# Reorder the DataFrame columns based on the sorted list
data_Versuche = data_Versuche.reindex(columns=sorted_columns)

# Extract columns with the same last 3 characters
columns_by_suffix = {}
for col in data_Versuche.columns:
    suffix = col.split()[-1]  # Extract last part of the column name
    if suffix not in columns_by_suffix:
        columns_by_suffix[suffix] = []
    columns_by_suffix[suffix].append(col)

# Calculate the difference for each unique pair of columns
differences = {}
for suffix, cols in columns_by_suffix.items():
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col1 = cols[i]
            col2 = cols[j]
            diff_col_name = f"{col1} - {col2}"  # New column name for the difference
            differences[diff_col_name] = data_Versuche[col1] - data_Versuche[col2]

# Combine the differences into a new DataFrame
data_diff = pd.DataFrame(differences)

print(data_diff)

## Erstellen der Export Ordner
def ensure_folder_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path
export_folder = ensure_folder_exists(os.path.join(current_directory, 'Export'))
file_export_folder = ensure_folder_exists(os.path.join(export_folder, folder_name))
plots_folder = ensure_folder_exists(os.path.join(file_export_folder, 'Plots'))
sensoren_folder = ensure_folder_exists(os.path.join(plots_folder, 'Sensoren'))
versuche_folder = ensure_folder_exists(os.path.join(plots_folder, 'Versuche'))

def format_group_label(group):
    if isinstance(group, tuple):
        return f"{group[0]} ({group[1]}, {group[2]})"
    return str(group)

## Ausreißer 1 plotten
for column in data_10min.columns[:]:  # Überspringe den ersten Index, falls das die Zeitstempel enthält
    outliers = data_outliers[f'{column}_Outlier']  # Hole die Ausreißer für die aktuelle Spalte

    # Extract group and format title
    group_label = format_group_label(column)

    plt.figure(figsize=(10, 6))
    # Punktdiagramm der Daten
    plt.plot(data_10min.index, data_10min[column], label='Data', color='blue', linewidth=2)
    # Markiere die Ausreißer
    plt.scatter(data_10min.index[outliers], data_10min[column][outliers], color='red', label='Outliers', s=20, zorder=5)

    # Titel und Achsenbeschriftungen
    plt.xlabel('Timestamp')
    plt.ylabel(f'Temperature')
    plt.title(f'{group_label}')
    plt.legend()

    # Diagramm speichern
    plot_file = os.path.join(sensoren_folder, f'{group_label}.png')
    plt.savefig(plot_file)
    plt.close()  # Schließe die aktuelle Figur, um Speicher zu sparen

print(f"Plots wurden im Ordner '{plots_folder}' gespeichert.")

## Ausreißer 2 plotten
for column in data_Versuche.columns:
    # Extract corresponding original data
    original_data_group = data_hourly[groups[column]]
    cleaned_data = data_Versuche[column]

    # Plot all data
    plt.figure(figsize=(12, 6))
    # Plot each column of original data
    for col in original_data_group.columns:
        col_label = format_group_label(col)
        plt.plot(data_hourly.index, original_data_group[col], label=f'Data ({col_label})', alpha=0.6)

    # Plot cleaned data (group mean remains as a single line)
    plt.plot(data_hourly.index, cleaned_data, label='Cleaned Data', color='blue', linewidth=2)

    # Highlight outliers only if they exist in data_unusual_2
    for group_column in groups[column]:
        if group_column in data_unusual_2.columns:
            outliers_group = data_unusual_2[group_column]

            # Ensure both arrays have the same index
            aligned_outliers_group = outliers_group.reindex(data_hourly.index)

            # Plot the aligned outliers
            outlier_label = format_group_label(group_column)
            plt.scatter(data_hourly.index, aligned_outliers_group, label=f'Outliers ({outlier_label})', alpha=0.8, zorder=5, s=20)

    # Add labels, legend, and title
    plt.xlabel('Timestamp')
    plt.ylabel(f'Temperature')
    plt.title(f'{column}')
    plt.legend(loc='upper left', fontsize='small', ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save the plot to the "Plots" folder
    plot_path = os.path.join(versuche_folder, f'{column}.png')
    plt.savefig(plot_path)
    plt.close()  # Close the plot to save memory

data_unusual.dropna(how='all', inplace=True)  # Löschen leerer Zeilen
data_unusual = data_unusual.dropna(axis=1, how='all')  # Löschen leerer Spalten
data_unusual.reset_index(inplace=True)
count_row = pd.DataFrame([data_unusual.count()], columns=data_unusual.columns)
data_unusual = pd.concat([count_row, data_unusual], ignore_index=True)  # Anzahl der Ausreißer als extra Reihe
data_unusual['Ausreißer gesamt'] = np.nan  # neue Spalte
data_unusual.loc[0, 'Timestamp'] = pd.NaT  # gesamte Spalte als Timestamp
data_unusual.loc[0, 'Ausreißer gesamt'] = data_unusual.iloc[0].sum()
print("Ausreißer:")
print(data_unusual)

data_unusual_2.dropna(how='all', inplace=True)  # Löschen leerer Zeilen
data_unusual_2 = data_unusual_2.dropna(axis=1, how='all')  # Löschen leerer Spalten
data_unusual_2.reset_index(inplace=True)
count_row = pd.DataFrame([data_unusual_2.count()], columns=data_unusual_2.columns)
data_unusual_2 = pd.concat([count_row, data_unusual_2], ignore_index=True)  # Anzahl der Ausreißer als extra Reihe
data_unusual_2['Ausreißer gesamt'] = np.nan  # neue Spalte
data_unusual_2.loc[0, 'Timestamp'] = pd.NaT  # gesamte Spalte als Timestamp
data_unusual_2.loc[0, 'Ausreißer gesamt'] = data_unusual_2.iloc[0].sum()
print("Ausreißer 2:")
print(data_unusual_2)

## Export der Daten
dataframes_to_export = [  # Liste des Exports
    (data_hourly, '_Sensoren.csv'),
    (data_unusual, '_unusual.csv'),
    (data_Versuche, '_Versuche.csv'),
    (data_Versuche_min_max, '_min_max.csv'),
    (data_unusual_2, '_unusual_2.csv'),
    (data_diff, '_Differenz.csv')
]

def simplify_column_name(col):  # Vorbereitung Namen
    if isinstance(col, tuple):
        return ", ".join(str(item) for item in col)
    return str(col)


for df, _ in dataframes_to_export:  # Export Funktion
    df.columns = [simplify_column_name(col) for col in df.columns]
    df.reset_index(inplace=True)

    # Prüfen, ob die Spalte 'Timestamp' vorhanden ist
    if 'Timestamp' in df.columns:
        # Konvertiere Timestamp in datetime-Objekte
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

        # Prüfen, ob die Zeitstempel tz-naiv sind
        if df['Timestamp'].dt.tz is None:
            # Zeitzone lokalisieren, wenn nicht vorhanden
            df['Timestamp'] = df['Timestamp'].dt.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT')

        # In die gewünschte Zeitzone umwandeln
        df['Timestamp'] = df['Timestamp'].dt.tz_convert('Europe/Berlin')

        # Formatieren des Zeitstempels
        df['Timestamp'] = df['Timestamp'].dt.strftime('%d.%m.%Y %H:%M')
    else:
        print("Warnung: Die Spalte 'Timestamp' ist nicht in der DataFrame vorhanden!")

# for df, suffix in dataframes_to_export:  # Export als .csv
#     new_file_name = os.path.splitext(os.path.basename(file_path))[0] + suffix
#     new_file_path = os.path.join(file_export_folder, new_file_name)
#     df.to_csv(new_file_path, index=False)
#     print(f"Data exported to 'Export/{new_file_name}' successfully.")

excel_file_name = os.path.splitext(os.path.basename(file_path))[0] + '_all_data.xlsx'  # Export als .xlsx
excel_file_path = os.path.join(file_export_folder, excel_file_name)
with pd.ExcelWriter(excel_file_path, engine='xlsxwriter') as writer:
    for df, sheet_name in dataframes_to_export:
        # Write each dataframe to a different sheet
        df.to_excel(writer, sheet_name=sheet_name, index=False)
print(f"Data exported to 'Export/{excel_file_name}' successfully.")

print('done')