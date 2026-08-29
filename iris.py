import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Set plot style
sns.set_theme(style="whitegrid")

# ----------------------------------------------------
# 1. LOAD AND EXPLORE DATASET
# ----------------------------------------------------
# Load the CSV file
df = pd.read_csv("Iris.csv")

# Drop the 'Id' column if present
if "Id" in df.columns:
    df.drop(columns=["Id"], inplace=True)

# Standardize column names
df.columns = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species",
]

print("--- Dataset Preview ---")
print(df.head())
print("\n--- Class Distribution ---")
print(df["species"].value_counts())

# ----------------------------------------------------
# 2. DATA VISUALIZATION
# ----------------------------------------------------
# Pairplot to observe feature separability
sns.pairplot(df, hue="species", markers=["o", "s", "D"], palette="Set2")
plt.suptitle("Pairwise Relationships of Iris Features", y=1.02)
plt.show()

# ----------------------------------------------------
# 3. DATA PREPARATION & SPLIT
# ----------------------------------------------------
# Features (X) and Target (y)
X = df.drop(columns=["species"])
y = df["species"]

# 80-20 Train-Test Split with stratification to keep equal class ratios
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize feature values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------
# 4. MODEL TRAINING (K-Nearest Neighbors)
# ----------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# ----------------------------------------------------
# 5. EVALUATION
# ----------------------------------------------------
# Make predictions on the test set
y_pred = knn.predict(X_test_scaled)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%\n")
print("--- Classification Report ---")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
labels = df["species"].unique()

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
)
plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# ----------------------------------------------------
# 6. SINGLE SAMPLE PREDICTION DEMO
# ----------------------------------------------------
# Sample measurements: [sepal_length, sepal_width, petal_length, petal_width]
sample_measurement = [[5.1, 3.5, 1.4, 0.2]]
sample_scaled = scaler.transform(sample_measurement)
prediction = knn.predict(sample_scaled)

print(f"Sample Input {sample_measurement} -> Predicted Species: {prediction[0]}")
