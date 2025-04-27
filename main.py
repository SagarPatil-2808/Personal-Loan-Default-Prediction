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

st.set_page_config(page_title="Loan Default Prediction", layout="wide")

# 1. Load & clean data
df = pd.read_csv('Loan_default.csv')
df.columns = df.columns.str.strip()

# 2. Split into train/test
target = 'Default'
X = df.drop(columns=[target])
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Identify types
numeric_cols = X.select_dtypes(['int64','float64']).columns.tolist()
cat_cols     = X.select_dtypes(['object']).columns.tolist()

# 4. Build preprocessing pipeline
num_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale',  StandardScaler())
])
cat_pipe = Pipeline([
    ('impute',   SimpleImputer(strategy='most_frequent')),
    ('onehot',   OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', num_pipe, numeric_cols),
    ('cat', cat_pipe, cat_cols)
], remainder='drop')

# 5. Define model pool
models = {
    'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost':             xgb.XGBClassifier(
                              n_estimators=100,
                              use_label_encoder=False,
                              eval_metric='logloss',
                              random_state=42
                           )
}

# 6. Sidebar: choose model
page = st.sidebar.radio("Select Model", list(models.keys()))

# 7. Assemble & train selected model
pipe = Pipeline([
    ('preproc', preprocessor),
    ('clf',     models[page])
])

with st.spinner(f"Training {page} — please wait…"):
    pipe.fit(X_train, y_train)

# 8. Evaluate on hold-out
y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:, 1]

metrics = {
    'Precision': precision_score(y_test,   y_pred),
    'Recall':    recall_score(y_test,      y_pred),
    'AUC-ROC':   roc_auc_score(y_test, y_proba)
}

# 9. Display performance
st.title(f"Loan Default Prediction — {page}")
perf_df = pd.DataFrame(metrics, index=[page]).T
perf_df.index.name = 'Metric'
st.subheader("Model Performance")
st.table(perf_df)

# 10. Prediction form
st.subheader("Predict Applicant’s Default Risk")
st.write("Enter applicant details:")

with st.form('input_form'):
    inputs = {}
    cols = numeric_cols + cat_cols
    half = len(cols) // 2
    c1, c2 = st.columns(2)
    for col in cols[:half]:
        if col in numeric_cols:
            inputs[col] = c1.number_input(col, value=float(df[col].median()))
        else:
            inputs[col] = c1.selectbox(col, df[col].unique())
    for col in cols[half:]:
        if col in numeric_cols:
            inputs[col] = c2.number_input(col, value=float(df[col].median()))
        else:
            inputs[col] = c2.selectbox(col, df[col].unique())
    submitted = st.form_submit_button("Predict")

if submitted:
    inp_df = pd.DataFrame([inputs])
    score = pipe.predict_proba(inp_df)[0,1]
    st.write(f"**Default Probability:** {score:.2%}")
    if score < 0.5:
        st.success("✅ Low Risk: Applicant is likely eligible for the loan.")
    else:
        st.error("⚠️ High Risk: Applicant may default. Further review needed.")
