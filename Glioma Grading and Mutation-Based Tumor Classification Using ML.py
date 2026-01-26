import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

file_path = "TCGA_GBM_LGG_Mutations_all.csv"
df = pd.read_csv(file_path)

df['Age_at_diagnosis'] = pd.to_numeric(df['Age_at_diagnosis'], errors='coerce') / 365

mutation_cols = ['IDH1', 'TP53', 'ATRX', 'EGFR', 'PTEN']

grade_counts = df['Grade'].value_counts()
plt.figure(figsize=(6,4))
grade_counts.plot(kind='bar')
plt.title('Distribution of Glioma Grades')
plt.xlabel('Grade (0 = LGG, 1 = GBM)')
plt.ylabel('Count')
plt.grid(True)
plt.show()

gender_counts = df['Gender'].value_counts()
plt.figure(figsize=(6,4))
gender_counts.plot(kind='bar')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
plt.hist(df['Age_at_diagnosis'].dropna(), bins=30, edgecolor='black')
plt.title('Age at Diagnosis Distribution')
plt.xlabel('Age (Years)')
plt.ylabel('Patients')
plt.grid(True)
plt.show()

race_counts = df['Race'].value_counts()
plt.figure(figsize=(8,5))
race_counts.plot(kind='bar')
plt.title('Race Distribution')
plt.xlabel('Race')
plt.ylabel('Count')
plt.grid(True)
plt.show()

avg_age = df.groupby('Grade')['Age_at_diagnosis'].mean()
plt.figure(figsize=(6,4))
avg_age.plot(kind='bar')
plt.title('Average Age by Grade')
plt.xlabel('Grade')
plt.ylabel('Average Age')
plt.grid(True)
plt.show()

idh1_dist = df.groupby(['Grade','IDH1']).size().unstack(fill_value=0)
idh1_dist.plot(kind='bar', stacked=True)
plt.title('IDH1 Mutation Distribution by Grade')
plt.xlabel('Grade')
plt.ylabel('Count')
plt.grid(True)
plt.show()

tp53_counts = df['TP53'].value_counts()
plt.figure(figsize=(6,6))
plt.pie(tp53_counts, labels=['Not Mutated','Mutated'], autopct='%1.1f%%')
plt.title('TP53 Mutation Status')
plt.show()

gbm = df[df['Grade']==1][mutation_cols].mean()
lgg = df[df['Grade']==0][mutation_cols].mean()

plt.figure(figsize=(12,5))
plt.plot(gbm.values, label='GBM')
plt.plot(lgg.values, label='LGG')
plt.xticks(np.arange(len(mutation_cols)), mutation_cols, rotation=90)
plt.ylabel('Mutation Frequency')
plt.title('Gene Mutation Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

gender_age = df.groupby('Gender')['Age_at_diagnosis'].mean()
plt.figure(figsize=(6,4))
gender_age.plot(kind='bar')
plt.title('Average Age by Gender')
plt.xlabel('Gender')
plt.ylabel('Average Age')
plt.grid(True)
plt.show()

race_grade = pd.crosstab(df['Race'], df['Grade'])
race_grade.plot(kind='bar', stacked=True)
plt.title('Grade Distribution across Races')
plt.xlabel('Race')
plt.ylabel('Count')
plt.grid(True)
plt.tight_layout()
plt.show()

features = ['Age_at_diagnosis','IDH1','TP53','ATRX','EGFR','PTEN']
df_ml = df[features + ['Grade']].dropna()

X = df_ml[features]
y = df_ml['Grade']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))
