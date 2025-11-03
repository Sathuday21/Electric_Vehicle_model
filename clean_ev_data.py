import pandas as pd
import sys

# --- 1. Load Original Data ---
input_filename = 'electric_vehicle_analytics.csv'
output_filename = 'ev_processed.csv'

try:
    df = pd.read_csv(input_filename)
    print(f"Successfully loaded '{input_filename}'")
except FileNotFoundError:
    print(f"--- ERROR: '{input_filename}' not found. ---")
    print("Please make sure your dataset is in the same folder as this script.")
    sys.exit() # Stop the script
except Exception as e:
    print(f"An error occurred loading the file: {e}")
    sys.exit()

print("\n--- Initial Data Info (Before Cleaning) ---")
df.info()

# --- 2. Data Cleaning ---
print("\n--- Cleaning Data ---")

# Drop columns that are not useful for prediction
# 'Model' has too many unique values, 'Vehicle_ID' is just an identifier
df_processed = df.drop(['Vehicle_ID', 'Model'], axis=1)

# Define the text columns we need to convert
categorical_cols = ['Make', 'Region', 'Vehicle_Type', 'Usage_Type']

# Convert text columns to numbers using one-hot encoding
df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)

print("Finished dropping columns and one-hot encoding.")

# --- 3. Save Cleaned Data ---
df_processed.to_csv(output_filename, index=False)

print("\n--- Data Cleaning Complete! ---")
print(f"Cleaned data has been saved as '{output_filename}'")