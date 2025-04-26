# 🚀 Loan Default Prediction App

Welcome to the **Loan Default Prediction App**, a Streamlit-based web application that predicts whether a loan applicant is likely to default based on key financial and personal details.

🔗 **Live Demo:** [Personal Loan Default Prediction App](https://personal-loan-default-prediction.streamlit.app/)

---

## 🌟 Features

- **Interactive EDA Dashboard**  
  - View raw data samples and missing-value summary  
  - Explore target class balance  
  - Inspect feature correlations through a heatmap  

- **Model Training & Evaluation**  
  - Train **Logistic Regression**, **Random Forest**, and **XGBoost** models  
  - Evaluate each model on **Precision**, **Recall**, and **AUC–ROC**  
  - View performance metrics in an easy-to-read table  

- **Model-Specific Pages**  
  - Each model has its dedicated page showing only that model’s metrics and prediction tool  

- **Real-Time Prediction**  
  - Enter applicant details via an intuitive two-column form  
  - Receive instant default-risk probability and loan-eligibility verdict  

---

## 🛠️ Usage

Run the Streamlit app:
```bash
streamlit run main.py
```

Then open your browser at `http://localhost:8501` to interact with the app.

---

## 📊 App Structure

```
├── data/
│   └── Loan_default.csv      # Raw dataset
├── main.py                   # Streamlit application script
├── requirements.txt          # Python dependencies
└── README.md                 # Project overview
```

---

## 🔍 Detailed Breakdown

### 1. Data Loading & Preprocessing
- **Missing-Value Imputation**: Median for numerical, most-frequent for categorical  
- **Scaling**: StandardScaler on numeric columns  
- **Encoding**: One-hot encoding for categorical features  

### 2. EDA & Visualization
- Raw data preview and missing-value summary  
- Target distribution bar chart  
- Heatmap of numeric-feature correlations  

### 3. Model Training
- **Logistic Regression**: Fast baseline  
- **Random Forest**: Ensemble tree-based model  
- **XGBoost**: Gradient boosting for high performance  

Metrics computed on test set:
- **Precision**  
- **Recall**  
- **AUC–ROC**  

### 4. Prediction Interface
- Two-column form with 7 features each for compact layout  
- Default probability displayed in percentage  
- Color-coded eligibility verdict  

---

## 📫 Contact

For questions or feedback, reach out to:

- **Your Name** – Sagar Patil  
- GitHub: [@SagarPatil-2808](https://github.com/SagarPatil-2808)

Enjoy exploring applicant risk profiles with clear, data-driven insights! 🎉
