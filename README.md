# 🧠 Glioma Grading: LGG vs GBM Classification

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)

---

## 📌 Project Overview

This project focuses on classifying brain tumors into two categories:

- **LGG** – Lower Grade Glioma (slow-growing)
- **GBM** – Glioblastoma Multiforme (aggressive, fast-growing)

Using **clinical and gene mutation data** from the TCGA (The Cancer Genome Atlas), we perform detailed **Exploratory Data Analysis (EDA)** and apply **Machine Learning models** to accurately predict tumor grade based on mutation patterns and patient demographics.

---

## 🧬 Dataset Information

| Property | Details |
|---|---|
| **File** | `TCGA_GBM_LGG_Mutations_all.xlsx` |
| **Source** | [TCGA – The Cancer Genome Atlas](https://www.cancer.gov/ccg/research/genome-sequencing/tcga) |
| **Target Variable** | `Grade` → `0 = LGG`, `1 = GBM` |

### Features Used

| Feature | Description |
|---|---|
| `Age_at_diagnosis` | Patient age (converted from days → years) |
| `IDH1` | IDH1 gene mutation status |
| `TP53` | TP53 gene mutation status |
| `ATRX` | ATRX gene mutation status |
| `EGFR` | EGFR gene mutation status |
| `PTEN` | PTEN gene mutation status |

---

## 📊 Exploratory Data Analysis (EDA)

The project includes rich visualizations to uncover patterns:

- 📊 Grade distribution (LGG vs GBM)
- 👨‍⚕️ Gender distribution by grade
- 🎂 Age distribution across grades
- 🌍 Race distribution
- 📉 Average age by tumor grade
- 🧬 Mutation frequency comparison (LGG vs GBM)
- 🔬 Mutation distribution per grade

> Key Insight: GBM patients tend to be older and show distinct mutation patterns in genes like EGFR and PTEN compared to LGG patients.

---

## ⚙️ Machine Learning Models

### 1️⃣ Logistic Regression
- Binary classification (LGG vs GBM)
- `max_iter = 1000`

### 2️⃣ K-Nearest Neighbors (KNN)
- `n_neighbors = 5`

Both models are evaluated and compared to identify the best performer on this biomedical dataset.

---

## 🧪 Data Preprocessing Pipeline

1. Converted `Age_at_diagnosis` from days → years
2. Dropped missing/null values
3. Encoded mutation status → binary (`1 = Mutated`, `0 = Not Mutated`)
4. Selected relevant features
5. Stratified train-test split **(80% train / 20% test)**
6. Feature scaling using **StandardScaler**

---

## 📈 Model Evaluation Metrics

Each model is evaluated using:

- ✅ Accuracy Score
- ✅ Confusion Matrix
- ✅ Classification Report (Precision, Recall, F1-Score)

---

## 📂 Project Structure

```
Glioma-Grading-Analysis/
│
├── giloma.py                          # Main Python script
├── TCGA_GBM_LGG_Mutations_all.xlsx    # Dataset
├── README.md                          # Project documentation
└── requirements.txt                   # Dependencies
```

---

## ▶️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Sangam919/Gliomo-Grading-Brain-Tumor-LGG-GBM-Analysis.git
cd Gliomo-Grading-Brain-Tumor-LGG-GBM-Analysis
```

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib scikit-learn openpyxl
```

### 3. Run the Script

```bash
python giloma.py
```

---

## 🚀 Future Improvements

- [ ] Add Random Forest & XGBoost classifiers
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] ROC-AUC Curve visualization
- [ ] K-Fold Cross-Validation
- [ ] Deploy as a **Streamlit Web App**
- [ ] Implement Deep Learning (ANN/CNN)
- [ ] SHAP values for model explainability

---

## 🧠 Skills Demonstrated

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Logistic Regression
- K-Nearest Neighbors Classification
- Model Evaluation & Comparison
- Biomedical / Genomic Data Analysis

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👨‍💻 Author

**Sangam Srivastav**  
B.Tech CSE | Machine Learning Enthusiast | Data Science Aspirant

[![GitHub](https://img.shields.io/badge/GitHub-Sangam919-black?style=flat-square&logo=github)](https://github.com/Sangam919)

---

> ⭐ If you found this project helpful, please consider giving it a star on GitHub!
