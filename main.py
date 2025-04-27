import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, roc_auc_score

# Page layout
st.set_page_config(page_title="Loan Default Prediction", layout="wide")

# 1) Load & clean data
df = pd.read_csv('Loan_default.csv')
df.columns = df.columns.str.strip()

# 2) Split into features/target and then train/test
X = df.drop(columns=['Default'])
y = df['Default']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3) Identify numerical vs categorical features
numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols     = X.select_dtypes(include=['object']).columns.tolist()

# 4) Preprocessing pipelines
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot',  OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', num_pipe, numeric_cols),
    ('cat', cat_pipe, cat_cols)
], remainder='drop')

# 5) Compute weight for XGBoost to handle imbalance
negatives, positives = np.bincount(y_train)
scale_pos_weight = negatives / positives

# 6) Define the three balanced models
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=500, class_weight='balanced', random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=42
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=100,
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        random_state=42
    ),
}

# 7) Sidebar controls
model_name    = st.sidebar.radio("Select model", list(models.keys()))
auto_override = st.sidebar.checkbox("Auto-fail if CreditScore < 300 or DTI > 0.4", True)

# 8) Build & train the selected pipeline
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier',   models[model_name])
])
with st.spinner(f"Training {model_name}…"):
    pipe.fit(X_train, y_train)

# 9) Evaluate on the hold-out set
y_pred  = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:, 1]
metrics = {
    'Precision': precision_score(y_test, y_pred),
    'Recall':    recall_score(y_test, y_pred),
    'AUC-ROC':   roc_auc_score(y_test, y_proba)
}

st.title(f"Loan Default Prediction — {model_name}")
st.subheader("Model Performance")
perf_df = pd.DataFrame(metrics, index=[model_name]).T
perf_df.index.name = 'Metric'
st.table(perf_df)

# 10) Input form for new predictions
st.subheader("Predict Applicant’s Default Risk")
inputs = {}
cols   = numeric_cols + cat_cols
col1, col2 = st.columns(2)
half = len(cols) // 2

# Left column inputs
for col in cols[:half]:
    if col in numeric_cols:
        inputs[col] = col1.number_input(col, value=float(df[col].median()))
    else:
        inputs[col] = col1.selectbox(col, df[col].unique())

# Right column inputs
for col in cols[half:]:
    if col in numeric_cols:
        inputs[col] = col2.number_input(col, value=float(df[col].median()))
    else:
        inputs[col] = col2.selectbox(col, df[col].unique())

# Prediction button
if st.button("Predict"):
    inp_df = pd.DataFrame([inputs])
    prob = pipe.predict_proba(inp_df)[0, 1]

    # Business‐rule override
    if auto_override and (inp_df['CreditScore'][0] < 300 or inp_df['DTIRatio'][0] > 0.4):
        prob = 1.0

    st.write(f"**Default Probability:** {prob:.2%}")
    if prob >= 0.5:
        st.error("⚠️ High Risk: Applicant may default. Further review needed.")
    else:
        st.success("✅ Low Risk: Applicant is likely eligible for the loan.") 


