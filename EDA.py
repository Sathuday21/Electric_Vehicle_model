import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# --- 1. Load Cleaned Data ---
input_filename = 'ev_processed.csv'

try:
    df_processed = pd.read_csv(input_filename)
    print(f"Successfully loaded '{input_filename}'")
except FileNotFoundError:
    print(f"--- ERROR: '{input_filename}' not found. ---")
    print("Please run '1_data_cleaning.py' first to create this file.")
    sys.exit() # Stop the script
except Exception as e:
    print(f"An error occurred loading the file: {e}")
    sys.exit()

print("\n--- 2. Generating EDA Plots (plots will open in new windows) ---")

# --- a. Correlation Heatmap ---
# We'll focus on the main numeric features to keep the plot readable
main_numeric_features = [
    'Year', 'Battery_Capacity_kWh', 'Battery_Health_%', 'Range_km',
    'Charging_Power_kW', 'Charging_Time_hr', 'Charge_Cycles',
    'Energy_Consumption_kWh_per_100km', 'Mileage_km', 'Avg_Speed_kmh',
    'Max_Speed_kmh', 'Acceleration_0_100_kmh_sec', 'Temperature_C',
    'CO2_Saved_tons', 'Maintenance_Cost_USD', 'Insurance_Cost_USD',
    'Electricity_Cost_USD_per_kWh', 'Monthly_Charging_Cost_USD',
    'Resale_Value_USD'
]

print("Generating Plot 1: Correlation Heatmap...")
plt.figure(figsize=(12, 10))
corr_matrix = df_processed[main_numeric_features].corr()

# We plot only the correlation with 'Resale_Value_USD' for clarity
sns.heatmap(corr_matrix[['Resale_Value_USD']].sort_values(by='Resale_Value_USD', ascending=False), 
            annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation with Resale Value (USD)')
plt.tight_layout()
plt.show() # This command opens the plot window

# --- b. Scatter Plot: Price vs. Range ---
print("Generating Plot 2: Resale Value vs. Range...")
plt.figure(figsize=(8, 5))
plt.scatter(df_processed['Range_km'], df_processed['Resale_Value_USD'], alpha=0.5)
plt.title('Resale Value vs. Range')
plt.xlabel('Range (km)')
plt.ylabel('Resale Value ($)')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- c. Scatter Plot: Price vs. Battery Capacity ---
print("Generating Plot 3: Resale Value vs. Battery Capacity...")
plt.figure(figsize=(8, 5))
plt.scatter(df_processed['Battery_Capacity_kWh'], df_processed['Resale_Value_USD'], alpha=0.5)
plt.title('Resale Value vs. Battery Capacity')
plt.xlabel('Battery (kWh)')
plt.ylabel('Resale Value ($)')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- d. Scatter Plot: Price vs. Mileage ---
print("Generating Plot 4: Resale Value vs. Mileage...")
plt.figure(figsize=(8, 5))
plt.scatter(df_processed['Mileage_km'], df_processed['Resale_Value_USD'], alpha=0.5)
plt.title('Resale Value vs. Mileage')
plt.xlabel('Mileage (km)')
plt.ylabel('Resale Value ($)')
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n--- EDA Complete! ---")