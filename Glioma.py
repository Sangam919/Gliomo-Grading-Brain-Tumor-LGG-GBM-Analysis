import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)

# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_excel("TCGA_GBM_LGG_Mutations_all.xlsx")

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(df.head())
print("\nShape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nMissing Values:\n", df.isnull().sum())

# ============================================================
# 2. DATA CLEANING & PREPROCESSING
# ============================================================

# Convert Age from days to years
df['Age_at_diagnosis'] = df['Age_at_diagnosis'] / 365

# Drop rows with missing values
df = df.dropna()

# Encode mutation columns to binary (Mutated = 1, Not Mutated = 0)
mutation_cols = ['IDH1', 'TP53', 'ATRX', 'EGFR', 'PTEN']

for col in mutation_cols:
    df[col] = df[col].apply(lambda x: 1 if str(x).strip().lower() == 'mutated' else 0)

# Encode Target Variable (LGG = 0, GBM = 1)
le = LabelEncoder()
df['Grade'] = le.fit_transform(df['Grade'])

print("\nLabel Mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# ============================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# --- Grade Distribution ---
plt.figure(figsize=(6, 4))
df['Grade'].value_counts().plot(kind='bar', color=['steelblue', 'tomato'], edgecolor='black')
plt.xticks([0, 1], ['LGG', 'GBM'], rotation=0)
plt.title("Grade Distribution (LGG vs GBM)")
plt.xlabel("Grade")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# --- Gender Distribution ---
if 'Gender' in df.columns:
    plt.figure(figsize=(6, 4))
    df['Gender'].value_counts().plot(kind='pie', autopct='%1.1f%%',
                                      colors=['lightblue', 'lightpink'])
    plt.title("Gender Distribution")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()

# --- Age Distribution ---
plt.figure(figsize=(7, 4))
sns.histplot(df['Age_at_diagnosis'], bins=30, kde=True, color='mediumseagreen')
plt.title("Age Distribution of Patients")
plt.xlabel("Age (Years)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# --- Average Age by Grade ---
plt.figure(figsize=(6, 4))
df.groupby('Grade')['Age_at_diagnosis'].mean().plot(kind='bar',
                                                     color=['steelblue', 'tomato'],
                                                     edgecolor='black')
plt.xticks([0, 1], ['LGG', 'GBM'], rotation=0)
plt.title("Average Age by Grade")
plt.xlabel("Grade")
plt.ylabel("Average Age (Years)")
plt.tight_layout()
plt.show()

# --- Race Distribution ---
if 'Race' in df.columns:
    plt.figure(figsize=(8, 4))
    df['Race'].value_counts().plot(kind='bar', color='mediumpurple', edgecolor='black')
    plt.title("Race Distribution")
    plt.xlabel("Race")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# --- Mutation Frequency Comparison (LGG vs GBM) ---
lgg = df[df['Grade'] == 0]
gbm = df[df['Grade'] == 1]

mut_lgg = lgg[mutation_cols].mean()
mut_gbm = gbm[mutation_cols].mean()

x = np.arange(len(mutation_cols))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width/2, mut_lgg, width, label='LGG', color='steelblue', edgecolor='black')
ax.bar(x + width/2, mut_gbm, width, label='GBM', color='tomato', edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(mutation_cols)
ax.set_title("Mutation Frequency Comparison: LGG vs GBM")
ax.set_ylabel("Mutation Rate")
ax.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 4. FEATURE & TARGET SPLIT
# ============================================================

features = ['Age_at_diagnosis'] + mutation_cols

X = df[features]
y = df['Grade']

print("\nFeatures used:", features)
print("X shape:", X.shape)
print("y distribution:\n", y.value_counts())

# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

# ============================================================
# 6. FEATURE SCALING
# ============================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ============================================================
# 7. MODEL 1 – LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("MODEL 1: LOGISTIC REGRESSION")
print("=" * 60)

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)

lr_pred = lr_model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, lr_pred))
print("\nClassification Report:\n", classification_report(y_test, lr_pred,
                                                           target_names=['LGG', 'GBM']))

# Confusion Matrix – LR
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, lr_pred), annot=True, fmt='d',
            cmap='Blues', xticklabels=['LGG', 'GBM'], yticklabels=['LGG', 'GBM'])
plt.title("Confusion Matrix – Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ============================================================
# 8. MODEL 2 – K-NEAREST NEIGHBORS
# ============================================================

print("\n" + "=" * 60)
print("MODEL 2: K-NEAREST NEIGHBORS (KNN)")
print("=" * 60)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

knn_pred = knn_model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, knn_pred))
print("\nClassification Report:\n", classification_report(y_test, knn_pred,
                                                           target_names=['LGG', 'GBM']))

# Confusion Matrix – KNN
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, knn_pred), annot=True, fmt='d',
            cmap='Oranges', xticklabels=['LGG', 'GBM'], yticklabels=['LGG', 'GBM'])
plt.title("Confusion Matrix – KNN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ============================================================
# 9. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

results = {
    "Logistic Regression": accuracy_score(y_test, lr_pred),
    "KNN"                : accuracy_score(y_test, knn_pred),
}

for model_name, acc in results.items():
    print(f"  {model_name}: {acc:.4f}")

plt.figure(figsize=(6, 4))
plt.bar(results.keys(), results.values(), color=['steelblue', 'tomato'], edgecolor='black')
plt.ylim(0, 1)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.show()

# ============================================================
# 10. SAVE MODELS
# ============================================================

import joblib

joblib.dump(lr_model,  "logistic_regression_model.pkl")
joblib.dump(knn_model, "knn_model.pkl")
joblib.dump(scaler,    "scaler.pkl")

print("\nModels and Scaler saved successfully!")
print("Files: logistic_regression_model.pkl | knn_model.pkl | scaler.pkl")
