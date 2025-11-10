import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import sys
import joblib # Used to save your model


warnings.filterwarnings('ignore')

# ---  Load CLEANED Data ---
filename = 'ev_processed.csv'
try:
    df_processed = pd.read_csv(filename)
    print(f"Successfully loaded '{filename}'")
except FileNotFoundError:
    print(f"--- ERROR: File Not Found ---")
    print(f"The file '{filename}' was not found.")
    print("Please run '1_data_cleaning.py' first to create this file.")
    sys.exit() # Stop the script
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit()


#  Define Features (X) and Target (y) 
print("\n--- 2. Defining Features and Target ---")
# The target (y) is what you want to predict
y = df_processed['Resale_Value_USD']

# The features (X) are all other columns used to make the prediction
X = df_processed.drop('Resale_Value_USD', axis=1)

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")


#  Split Data into Training and Testing Sets
print("\n--- 3. Splitting Data ---")
# We use 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Split data: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples.")


#  Feature Scaling 
print("\n--- 4. Scaling Features ---")
# This scales all features to a similar range, which is important
scaler = StandardScaler()

# Fit the scaler ONLY on the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform both training and testing data
X_test_scaled = scaler.transform(X_test)
print("Feature scaling complete.")


#  5. Initialize, Train, and Evaluate Models 
print("\n--- 5. Training and Evaluating Models ---")

# We will test the 3 models from your project plan:
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
}

# Store results
results = {}
# Store the actual trained model objects
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model # Save the trained model
    
    # Make predictions on the unseen test data
    y_pred = model.predict(X_test_scaled)
    
    # Calculate evaluation metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Store and print results
    results[name] = {'R2': r2, 'MAE': mae}
    
    print(f"--- Results for {name} ---")
    print(f"  R-squared (R²): {r2:.4f}")
    print(f"  Mean Absolute Error (MAE): ${mae:,.2f}")


#  Compare Models and Save the Best One 
print("\n\n--- 6. Final Model Comparison & Saving ---")

# Find the best model based on R-squared (higher is better)
best_model_name = max(results, key=lambda name: results[name]['R2'])
best_model_object = trained_models[best_model_name]
best_r2 = results[best_model_name]['R2']
best_mae = results[best_model_name]['MAE']

print(f"Best Model (based on R-squared): {best_model_name}")
print(f"  Best R-squared: {best_r2:.4f}")
print(f"  Best MAE: ${best_mae:,.2f}")

# This is the "build" step: saving the final model 
# Save the model object
model_filename = 'final_model.pkl'
joblib.dump(best_model_object, model_filename)
print(f"\nSuccessfully saved the best model as '{model_filename}'")

# We MUST also save the scaler to process user input in the frontend
scaler_filename = 'scaler.pkl'
joblib.dump(scaler, scaler_filename)
print(f"Successfully saved the scaler as '{scaler_filename}'")


print("\n--- ML Model Building  Complete! ---")