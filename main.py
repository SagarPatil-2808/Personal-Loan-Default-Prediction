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

st.set_page_config(page_title="Loan Default Prediction App", layout="wide")

# Load and cache data
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv('Loan_default.csv')
    df.columns = df.columns.str.strip()
    return df

# Train and cache models
auth_models = ['Logistic Regression', 'Random Forest', 'XGBoost']
@st.cache_resource(show_spinner=True)
def train_models(df):
    target = 'Default'
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, numeric_cols),
        ('cat', cat_pipe, cat_cols)
    ], remainder='drop')
    models = {
        'Logistic Regression': LogisticRegression(max_iter=500),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    pipelines = {}
    metrics = {}
    for name, clf in models.items():
        pipe = Pipeline([('preproc', preprocessor), ('clf', clf)])
        pipe.fit(X_train, y_train)
        pipelines[name] = pipe
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:,1]
        metrics[name] = {
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'AUC-ROC': roc_auc_score(y_test, y_proba)
        }
    return pipelines, metrics, numeric_cols, cat_cols

# Main app
df = load_data()
pipelines, metrics, numeric_cols, cat_cols = train_models(df)

# Sidebar navigation
def_pages = ['Logistic Regression', 'Random Forest', 'XGBoost']
page = st.sidebar.radio("Select Page", def_pages)

st.header(f"{page} Performance and Prediction")
# Show performance metrics for selected model
model_metrics = metrics[page]
perf_df = pd.DataFrame(model_metrics, index=[page]).T
perf_df.index.name = 'Metric'
st.subheader("Model Performance")
st.table(perf_df)

st.subheader("Predict Default Risk & Eligibility")
st.write("Provide applicant details:")

with st.form('input_form'):
    inputs = {}
    col1, col2 = st.columns(2)
    all_features = numeric_cols + cat_cols
    col1_fields = all_features[:7]
    col2_fields = all_features[7:]

    with col1:
        for col in col1_fields:
            if col in numeric_cols:
                inputs[col] = st.number_input(col, value=float(df[col].median()))
            else:
                inputs[col] = st.selectbox(col, df[col].unique())
    with col2:
        for col in col2_fields:
            if col in numeric_cols:
                inputs[col] = st.number_input(col, value=float(df[col].median()))
            else:
                inputs[col] = st.selectbox(col, df[col].unique())
    submitted = st.form_submit_button('Predict')

if submitted:
    input_df = pd.DataFrame([inputs])
    model = pipelines[page]
    proba = model.predict_proba(input_df)[:,1][0]
    st.write(f"**Default Probability:** {proba:.2%}")
    if proba < 0.5:
        st.success("✅ Low Risk: Applicant is likely eligible for the loan.")
    else:
        st.error("⚠️ High Risk: Applicant may default. Further review needed.")


