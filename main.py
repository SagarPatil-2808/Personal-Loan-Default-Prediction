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

# 1) Load data
df = pd.read_csv('Loan_default.csv')
df.columns = df.columns.str.strip()

# 2) Split out target
X = df.drop(columns=['Default'])
y = df['Default']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3) Col types
num_cols = X.select_dtypes(['int64','float64']).columns.tolist()
cat_cols = X.select_dtypes(['object']).columns.tolist()

# 4) Preprocessor
num_pipe = Pipeline([('impute', SimpleImputer('median')), ('scale', StandardScaler())])
cat_pipe = Pipeline([('impute', SimpleImputer('most_frequent')), ('oh', OneHotEncoder(handle_unknown='ignore'))])
preproc   = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)], remainder='drop')

# 5) Compute XGBoost weight
neg, pos = np.bincount(y_train)
scale_pos_weight = neg/pos

# 6) Define models with balancing
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
page     = st.sidebar.radio("Select model", list(models.keys()))
thr      = st.sidebar.slider("High-risk threshold", 0.01, 0.99, 0.50)
auto_cr  = st.sidebar.checkbox("Auto-fail if CreditScore<300 or DTI>0.4", True)

# 8) Build, train
pipe = Pipeline([('prep', preproc), ('clf', models[page])])
with st.spinner(f"Training {page}..."):
    pipe.fit(X_train, y_train)

# 9) Eval on hold-out
y_p   = pipe.predict(X_test)
y_pr  = pipe.predict_proba(X_test)[:,1]
metrics = {
    'Precision': precision_score(y_test, y_p),
    'Recall':    recall_score(y_test, y_p),
    'AUC-ROC':   roc_auc_score(y_test, y_pr)
}

st.title(f"Loan Default Prediction — {page}")
st.subheader("Model Performance")
st.table(pd.DataFrame(metrics, index=[page]).T)

# 10) Prediction form
st.subheader("Predict Applicant's Default Risk")
inputs = {}
cols = num_cols + cat_cols
c1, c2 = st.columns(2)
for col in cols[:len(cols)//2]:
    inputs[col] = c1.number_input(col, value=float(df[col].median())) if col in num_cols else c1.selectbox(col, df[col].unique())
for col in cols[len(cols)//2:]:
    inputs[col] = c2.number_input(col, value=float(df[col].median())) if col in num_cols else c2.selectbox(col, df[col].unique())

if st.button("Predict"):
    inp = pd.DataFrame([inputs])
    prob = pipe.predict_proba(inp)[0,1]

    # business override
    if auto_cr and (inp['CreditScore'][0] < 300 or inp['DTIRatio'][0] > 0.4):
        prob = 1.0

    st.write(f"**Default Probability:** {prob:.2%}")
    if prob >= thr:
        st.error("⚠️ High Risk: Applicant may default. Further review needed.")
    else:
        st.success("✅ Low Risk: Applicant is likely eligible for the loan.")
