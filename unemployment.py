import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set visualization style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# ----------------------------------------------------
# 1. LOAD DATASET & PREPROCESSING
# ----------------------------------------------------
# Load the dataset (update filename/path if necessary)
df = pd.read_csv("Unemployment in India.csv")

# Strip leading/trailing whitespaces from column names
df.columns = df.columns.str.strip()

# Check and drop null / empty rows
df.dropna(inplace=True)

# Standardize column names for easy access
df.rename(
    columns={
        "Region": "State",
        "Date": "Date",
        "Frequency": "Frequency",
        "Estimated Unemployment Rate (%)": "Unemployment_Rate",
        "Estimated Employed": "Employed",
        "Estimated Labour Participation Rate (%)": "Labour_Participation_Rate",
        "Area": "Area",
    },
    inplace=True,
)

# Convert Date column to datetime format
df["Date"] = pd.to_datetime(df["Date"].str.strip(), format="%d-%m-%Y")

# Extract Year, Month, and Month Name for seasonal analysis
df["Year"] = df["Date"].dt.year
df["Month_Num"] = df["Date"].dt.month
df["Month_Name"] = df["Date"].dt.strftime("%b")

print("Dataset Preview:")
print(df.head())
print("\nDataset Info:")
print(df.info())

# ----------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS & VISUALIZATIONS
# ----------------------------------------------------

# (A) Overall Monthly Trend (Impact of COVID-19 Lockdown in March-April 2020)
monthly_trend = (
    df.groupby("Date")["Unemployment_Rate"]
    .mean()
    .reset_index()
    .sort_values("Date")
)

plt.figure(figsize=(12, 5))
sns.lineplot(
    data=monthly_trend,
    x="Date",
    y="Unemployment_Rate",
    marker="o",
    color="crimson",
    linewidth=2.5,
)
plt.axvline(
    pd.to_datetime("2020-03-24"),
    color="black",
    linestyle="--",
    label="COVID-19 Lockdown (Mar 2020)",
)
plt.title(
    "National Average Unemployment Rate Trend (2019 - 2020)",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Date")
plt.ylabel("Average Unemployment Rate (%)")
plt.legend()
plt.tight_layout()
plt.show()

# (B) Rural vs Urban Unemployment Comparison
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x="Area",
    y="Unemployment_Rate",
    palette="Set2",
    hue="Area",
    legend=False,
)
plt.title(
    "Unemployment Rate Distribution: Rural vs Urban",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Area")
plt.ylabel("Unemployment Rate (%)")
plt.tight_layout()
plt.show()

# (C) Pre-COVID vs Post-COVID Impact Analysis
# Lockdown officially began in late March 2020
df["Period"] = df["Date"].apply(
    lambda x: "Pre-Lockdown (Before Apr 2020)"
    if x < pd.to_datetime("2020-04-01")
    else "Post-Lockdown (Apr 2020 Onwards)"
)

plt.figure(figsize=(10, 5))
sns.barplot(
    data=df,
    x="Period",
    y="Unemployment_Rate",
    hue="Area",
    palette="coolwarm",
    ci=None,
)
plt.title(
    "Pre-Lockdown vs. Post-Lockdown Unemployment Rates",
    fontsize=14,
    fontweight="bold",
)
plt.ylabel("Mean Unemployment Rate (%)")
plt.xlabel("")
plt.tight_layout()
plt.show()

# (D) Top 10 Most Affected States (By Mean Unemployment Rate)
top_states = (
    df.groupby("State")["Unemployment_Rate"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

plt.figure(figsize=(12, 6))
sns.barplot(
    data=top_states,
    x="Unemployment_Rate",
    y="State",
    palette="viridis",
    hue="State",
    legend=False,
)
plt.title(
    "Top 10 States with Highest Average Unemployment Rate",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Average Unemployment Rate (%)")
plt.ylabel("State")
plt.tight_layout()
plt.show()

# (E) Correlation Heatmap
plt.figure(figsize=(8, 5))
numeric_cols = df[
    ["Unemployment_Rate", "Employed", "Labour_Participation_Rate"]
]
sns.heatmap(
    numeric_cols.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
)
plt.title("Correlation Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
