import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Set plot aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# ----------------------------------------------------
# 1. LOAD & INSPECT DATASET
# ----------------------------------------------------
# Load the dataset (typically 'car data.csv' from the Kaggle repository)
df = pd.read_csv("car data.csv")

# Clean column headers
df.columns = df.columns.str.strip()

print("--- Dataset Head ---")
print(df.head())
print("\n--- Missing Values Check ---")
print(df.isnull().sum())

# ----------------------------------------------------
# 2. FEATURE ENGINEERING & PREPROCESSING
# ----------------------------------------------------
# Calculate vehicle age from manufacturing Year (reference year set to current context)
current_year = 2026
df["Vehicle_Age"] = current_year - df["Year"]
df.drop(columns=["Car_Name", "Year"], inplace=True)

# Encode categorical variables: Fuel_Type, Seller_Type, Transmission
# Using drop_first=True to avoid dummy variable trap (multicollinearity)
df_encoded = pd.get_dummies(df, drop_first=True)

print("\n--- Processed Dataset Columns ---")
print(df_encoded.columns.tolist())

# ----------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ----------------------------------------------------
# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df_encoded.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# ----------------------------------------------------
# 4. DATA SPLIT (TRAIN & TEST)
# ----------------------------------------------------
X = df_encoded.drop(columns=["Selling_Price"])
y = df_encoded["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------------------
# 5. MODEL TRAINING & COMPARISON
# ----------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=100, random_state=42
    ),
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results[name] = {"R2 Score": r2, "MAE": mae, "RMSE": rmse}

# Display metrics
results_df = pd.DataFrame(results).T
print("\n--- Model Performance Comparison ---")
print(results_df.round(4))

# ----------------------------------------------------
# 6. FEATURE IMPORTANCE & RESIDUAL VISUALIZATION
# ----------------------------------------------------
# Best Model: Random Forest
rf_model = models["Random Forest Regressor"]
y_pred_rf = rf_model.predict(X_test)

# Plot: Actual vs Predicted Prices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.scatterplot(x=y_test, y=y_pred_rf, ax=axes[0], color="royalblue")
axes[0].plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--",
    lw=2,
)
axes[0].set_title("Actual vs Predicted Selling Prices", fontweight="bold")
axes[0].set_xlabel("Actual Price (Lakhs)")
axes[0].set_ylabel("Predicted Price (Lakhs)")

# Plot: Feature Importances
feat_importances = pd.Series(
    rf_model.feature_importances_, index=X.columns
).sort_values(ascending=True)
feat_importances.plot(kind="barh", ax=axes[1], color="teal")
axes[1].set_title(
    "Feature Importance (Random Forest)", fontweight="bold"
)
axes[1].set_xlabel("Importance")

plt.tight_layout()
plt.show()

# ----------------------------------------------------
# 7. SAMPLE PREDICTION DEMO
# ----------------------------------------------------
# Predict price for a sample vehicle using the trained Random Forest model
sample_car = pd.DataFrame(
    [
        {
            "Present_Price": 9.54,
            "Kms_Driven": 43000,
            "Owner": 0,
            "Vehicle_Age": 13,
            "Fuel_Type_Diesel": 1,
            "Fuel_Type_Petrol": 0,
            "Seller_Type_Individual": 0,
            "Transmission_Manual": 1,
        }
    ]
)

predicted_price = rf_model.predict(sample_car)[0]
print(f"\nEstimated Resale Price for Sample Car: ₹{predicted_price:.2f} Lakhs")
