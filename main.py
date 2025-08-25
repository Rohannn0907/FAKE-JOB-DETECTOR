import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin


# ---------- Columns ----------
TEXT_FEATS = ['title','company_profile','description','requirements','benefits']

NUM_FEATS = [
    'len_description','len_requirements','len_benefits','len_company_profile',
    'description_url_cnt','requirements_url_cnt','benefits_url_cnt','company_profile_url_cnt',
    'description_email_cnt','requirements_email_cnt','benefits_email_cnt','company_profile_email_cnt',
    'description_phone_cnt','requirements_phone_cnt','benefits_phone_cnt','company_profile_phone_cnt',
    'salary_missing','salary_dash_cnt'
]

BOOL_FEATS = ['telecommuting','has_company_logo','has_questions']  
CAT_FEATS = ['employment_type','required_experience','required_education','industry','function']


# ---------- Helpers ----------
def present(cols, df_cols): 
    return [c for c in cols if c in df_cols]

def add_simple_feats(df):
    df = df.copy()
    for col in ['description','requirements','benefits','company_profile']:
        if col not in df.columns:
            continue
        s = df[col].fillna("")
        df[f'len_{col}'] = s.str.len()
        df[f'{col}_url_cnt'] = s.str.count(r'(http|www\.)', flags=re.I)
        df[f'{col}_email_cnt'] = s.str.count(r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}', flags=re.I)
        df[f'{col}_phone_cnt'] = s.str.count(r'\+?\d[\d\-\s]{7,}\d')

    if 'salary_range' in df.columns:
        sr = df['salary_range'].fillna("")
        df['salary_missing'] = (sr == "").astype(int)
        df['salary_dash_cnt'] = sr.str.count('-')
    return df

def join_text_columns(df, text_cols):
    return (df[text_cols].fillna("")
            .agg(lambda row: " \n ".join(row.values.astype(str)), axis=1))


# ---------- FIX: define a picklable selector ----------
def select_all_text(X):
    return X["__all_text__"]


# ---------- Preprocessor ----------
def make_preprocessor(df):
    return ColumnTransformer(
        transformers=[
            ("text", Pipeline([
                ("selector", FunctionTransformer(select_all_text, validate=False)),
                ("tfidf", TfidfVectorizer(
                    strip_accents="unicode",
                    lowercase=True,
                    stop_words="english",
                    max_features=10000,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=1.0
                ))
            ]), ['__all_text__']),
            ("num", "passthrough", present(NUM_FEATS, df.columns)),
            ("bool", "passthrough", present(BOOL_FEATS, df.columns)),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), present(CAT_FEATS, df.columns)),
        ],
        remainder="drop",
        sparse_threshold=1.0
    )


# ---------- Training ----------
def train_and_eval(df):
    df = add_simple_feats(df)
    y = df['fraudulent'].astype(int)
    df['__all_text__'] = join_text_columns(df, present(TEXT_FEATS, df.columns))
    preproc = make_preprocessor(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(
        max_iter=300,
        class_weight="balanced",
        solver="liblinear"
    )

    pipe = Pipeline([
        ("pre", preproc),
        ("clf", model)
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, preds, digits=4))
    print("ROC-AUC:", roc_auc_score(y_test, proba))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    return pipe


# ---------- Main ----------
if __name__ == "__main__":
    import joblib

    df = pd.read_csv("fake_job_postings.csv")
    model = train_and_eval(df)
    joblib.dump(model, "fake_job_detector.joblib")
    print("✅ Model training complete. Saved as fake_job_detector.joblib")
