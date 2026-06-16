# ==========================================================
# BLOOD CLASSIFICATION USING MACHINE LEARNING
# COMPLETE PROJECT IN A SINGLE CELL
# ==========================================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv("blood_classification.csv")

print("="*60)
print("DATASET OVERVIEW")
print("="*60)

print(df.head())
print("\nDataset Shape:", df.shape)

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows")
print(df.duplicated().sum())

# ==========================================================
# DATA CLEANING
# ==========================================================

df.drop_duplicates(inplace=True)

for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

print("\nCleaning Completed")

# ==========================================================
# STATISTICAL SUMMARY
# ==========================================================

print("\nStatistical Summary")
print(df.describe())

# ==========================================================
# EXPLORATORY DATA ANALYSIS
# ==========================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns

for column in numeric_columns:

    plt.figure(figsize=(6,4))

    sns.histplot(
        data=df,
        x=column,
        kde=True
    )

    plt.title(f"Distribution of {column}")
    plt.show()

# ==========================================================
# CORRELATION HEATMAP
# ==========================================================

plt.figure(figsize=(12,8))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()

# ==========================================================
# BOXPLOTS
# ==========================================================

for column in numeric_columns:

    plt.figure(figsize=(5,4))

    sns.boxplot(y=df[column])

    plt.title(f"Boxplot of {column}")

    plt.show()

# ==========================================================
# COUNT PLOTS
# ==========================================================

target_col = df.columns[-1]

plt.figure(figsize=(6,4))

sns.countplot(
    data=df,
    x=target_col
)

plt.title("Target Class Distribution")
plt.show()

# ==========================================================
# FEATURES AND TARGET
# ==========================================================

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# ==========================================================
# FEATURE SCALING
# ==========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ==========================================================
# MACHINE LEARNING MODELS
# ==========================================================

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=8,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=5
        )
}

# ==========================================================
# MODEL TRAINING
# ==========================================================

results = {}

for name, model in models.items():

    print("\n" + "="*60)
    print(name)
    print("="*60)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    results[name] = accuracy

    print(
        "Accuracy:",
        round(accuracy*100,2),
        "%"
    )

    print("\nClassification Report")

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    plt.figure(figsize=(5,4))

    sns.heatmap(
        confusion_matrix(
            y_test,
            y_pred
        ),
        annot=True,
        fmt='d',
        cmap='Blues'
    )

    plt.title(
        f'Confusion Matrix - {name}'
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.show()

# ==========================================================
# MODEL COMPARISON
# ==========================================================

comparison = pd.DataFrame({

    "Model": results.keys(),
    "Accuracy": results.values()

})

comparison = comparison.sort_values(
    by="Accuracy",
    ascending=False
)

print("\nMODEL COMPARISON")
print(comparison)

# ==========================================================
# ACCURACY VISUALIZATION
# ==========================================================

plt.figure(figsize=(8,5))

sns.barplot(
    x="Model",
    y="Accuracy",
    data=comparison
)

plt.title(
    "Machine Learning Model Comparison"
)

plt.xticks(rotation=15)

plt.show()

# ==========================================================
# BEST MODEL
# ==========================================================

best_model_name = max(
    results,
    key=results.get
)

print("\nBest Model:", best_model_name)

best_model = models[best_model_name]

# ==========================================================
# SAMPLE PREDICTION
# ==========================================================

sample = X.iloc[[0]]

sample_scaled = scaler.transform(sample)

prediction = best_model.predict(
    sample_scaled
)

print("\nSample Prediction")
print("Predicted Class:", prediction[0])

# ==========================================================
# SAVE MODEL
# ==========================================================

import joblib

joblib.dump(
    best_model,
    "blood_classification_model.pkl"
)

joblib.dump(
    scaler,
    "blood_scaler.pkl"
)

print("\nModel Saved Successfully")

# ==========================================================
# END OF PROJECT
# ==========================================================

print("\nBlood Classification Project Completed Successfully")