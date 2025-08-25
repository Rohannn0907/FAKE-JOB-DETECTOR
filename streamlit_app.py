import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import re
import requests
from bs4 import BeautifulSoup
from sklearn.metrics import confusion_matrix, roc_curve, auc

# ------------------------
# NEW: Custom CSS for colors
# ------------------------
def set_custom_style():
    st.markdown("""
        <style>
        .stApp {
            background-color: #e6f0ff; /* A very light, cool blue */
            color: #1a1a1a; /* Dark gray for main text */
        }
        .stSidebar {
            background-color: #d1e3ff; /* A slightly darker shade for sidebar */
            color: #1a1a1a;
        }
        .stButton>button {
            background-color: #4a90e2; /* A strong blue for buttons */
            color: white; /* White button text */
            border-color: #4a90e2;
        }
        .stButton>button:hover {
            background-color: #58a6ff; /* Lighter blue hover effect */
            color: white;
        }
        .stTextInput > div > div > input, .stTextArea > div > div > textarea {
            background-color: #d1e3ff; /* Light input backgrounds */
            color: #1a1a1a;
            border-color: #4a90e2;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #004d99; /* A deep blue for headers */
        }
        .stProgress > div > div > div > div {
            background-color: #4a90e2; /* Blue progress bar */
        }
        .st-d9 { /* This targets markdown headers */
            color: #004d99; /* Deep blue headers */
        }
        .st-br { /* Targets markdown text in headers */
            color: #004d99;
        }
        .st-da { /* Targets the metric value text */
            color: #4a90e2 !important;
        }
        .stAlert > div {
            color: #1a1a1a; /* Ensure text is readable in alerts */
        }
        .stAlert.warning > div {
            background-color: #ffc107;
            color: #1a1a1a;
        }
        .stTable {
            color: #1a1a1a;
        }
        </style>
        """, unsafe_allow_html=True)
    
# ------------------------
# Initialize Session State
# ------------------------
if 'job_data' not in st.session_state:
    st.session_state.job_data = {
        "title": "Software Engineer",
        "company_profile": "We are a global leader in software solutions...",
        "requirements": "Bachelor's degree in Computer Science, 2+ years of Python experience",
        "description": "We are looking for a motivated developer to join our growing team...",
        "benefits": "Health insurance, Paid time off, Flexible working hours",
    }
    st.session_state.url = ""

# ------------------------
# Define helper functions
# ------------------------
def select_all_text(X):
    return X["__all_text__"]

def add_simple_feats(df):
    df = df.copy()
    for col in ['description', 'requirements', 'benefits', 'company_profile']:
        if col not in df.columns:
            continue
        s = df[col].fillna("")
        df[f'len_{col}'] = s.str.len()
        df[f'{col}_url_cnt'] = s.str.count(r'(http|www\.)', flags=re.I)
        df[f'{col}_email_cnt'] = s.str.count(r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}', flags=re.I)
        df[f'{col}_phone_cnt'] = s.str.count(r'\+?\d[\d\-\s]{7,}\d')

    df['salary_range'] = None
    df['salary_missing'] = 1
    df['salary_dash_cnt'] = 0
    df['telecommuting'] = 0
    df['has_company_logo'] = 0
    df['has_questions'] = 0
    
    cat_cols = ['employment_type', 'required_experience', 'required_education', 'industry', 'function']
    for col in cat_cols:
        if col not in df.columns:
            df[col] = "Other"
    return df

def join_text_columns(df, text_cols):
    return (df[text_cols].fillna("")
            .agg(lambda row: " \n ".join(row.values.astype(str)), axis=1))

def scrape_linkedin_job(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    
    title = ""
    company_profile = ""
    description = ""
    requirements = ""
    benefits = ""

    title_element = soup.find('h1', class_='top-card-layout__title')
    if title_element:
        title = title_element.get_text(strip=True)

    company_element = soup.find('div', class_='top-card-layout__card')
    if company_element:
        company_link = company_element.find('a', class_='topcard__org-name-link')
        if company_link:
            company_profile = company_link.get_text(strip=True)
    
    description_section = soup.find('div', class_='description__text')
    if description_section:
        full_text = description_section.get_text(strip=True)
        if "Benefits" in full_text:
            description, benefits = full_text.split("Benefits", 1)
        else:
            description = full_text

        req_match = re.search(r'Requirements|Qualifications|Experience', description, re.I)
        if req_match:
            req_start_index = req_match.start()
            requirements = description[req_start_index:].strip()
            description = description[:req_start_index].strip()
        
    return {
        "title": title,
        "company_profile": company_profile,
        "description": description,
        "requirements": requirements,
        "benefits": benefits
    }

def explain_prediction(input_df):
    reasons = []
    keywords = ["immediate start", "no experience necessary", "earn money from home", "data entry"]
    full_text = input_df.iloc[0]['__all_text__'].lower()
    for kw in keywords:
        if kw in full_text:
            reasons.append(f"Contains suspicious keywords like '{kw}'.")

    if not input_df.iloc[0]['company_profile']:
        reasons.append("Company profile is missing or very brief.")

    if input_df.iloc[0]['description_email_cnt'] > 0 or input_df.iloc[0]['requirements_email_cnt'] > 0:
        reasons.append("Email addresses are included in the job description or requirements, which is unusual for a legitimate posting.")

    if input_df.iloc[0]['description_phone_cnt'] > 0 or input_df.iloc[0]['requirements_phone_cnt'] > 0:
        reasons.append("Phone numbers are included in the job description or requirements, which is unusual for a legitimate posting.")
        
    if input_df.iloc[0]['len_description'] < 50:
        reasons.append("The job description is unusually short and lacks detail.")

    if input_df.iloc[0]['salary_missing'] == 1:
        reasons.append("Salary information is missing.")
        
    return reasons if reasons else ["No obvious red flags were detected, but the model's confidence is low."]


def populate_from_url():
    if st.session_state.url:
        with st.spinner("Fetching and analyzing content from URL..."):
            try:
                scraped_data = scrape_linkedin_job(st.session_state.url)
                st.session_state.job_data = scraped_data
                st.success("Data successfully scraped. Please review and click 'Analyze' to get a prediction.")
            except requests.exceptions.RequestException as e:
                st.error(f"Error fetching URL: {e}. Please check the URL and your internet connection.")
            except Exception as e:
                st.error(f"An error occurred during scraping: {e}")
                
# ------------------------
# Load Model
# ------------------------
@st.cache_resource
def load_model():
    return joblib.load("fake_job_detector.joblib")

model = load_model()

# ------------------------
# Set custom styles at the beginning of the app
# ------------------------
set_custom_style()

# ------------------------
# Streamlit Config
# ------------------------
st.set_page_config(
    page_title="Fake Job Detector",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
st.sidebar.title("🕵️ Fake Job Detector")
st.sidebar.markdown("AI-powered tool to detect fraudulent job postings.")
st.sidebar.markdown("### ⚙️ Settings")
show_metrics = st.sidebar.checkbox("Show Model Metrics", value=True)
show_confusion = st.sidebar.checkbox("Show Confusion Matrix", value=False)
st.title("🚨 Fake Job Posting Detector")
st.markdown("""This app uses **Machine Learning** to predict whether a job posting is **Real (0)** or **Fake (1)**.""")

# ------------------------
# URL Input Section
# ------------------------
st.subheader("🌐 Analyze from URL")
st.markdown("**(This feature is optimized for LinkedIn job postings)**")
st.text_input("Enter LinkedIn Job Posting URL", key="url")
st.button("🔍 Fetch from URL", on_click=populate_from_url)

# ------------------------
# Manual/Combined Input Section
# ------------------------
st.subheader("✍️ Job Posting Details")
col1, col2 = st.columns(2)
with col1:
    st.session_state.job_data["title"] = st.text_input("Job Title", value=st.session_state.job_data["title"], key="title")
    st.session_state.job_data["company_profile"] = st.text_area("Company Profile", value=st.session_state.job_data["company_profile"], key="company_profile")
    st.session_state.job_data["requirements"] = st.text_area("Requirements", value=st.session_state.job_data["requirements"], key="requirements")
with col2:
    st.session_state.job_data["description"] = st.text_area("Job Description", value=st.session_state.job_data["description"], key="description")
    st.session_state.job_data["benefits"] = st.text_area("Benefits", value=st.session_state.job_data["benefits"], key="benefits")

if st.button("📈 Analyze Job Posting"):
    input_dict = st.session_state.job_data
    input_df = pd.DataFrame([input_dict])
    
    input_df = add_simple_feats(input_df)
    input_df['__all_text__'] = join_text_columns(input_df, ['title', 'company_profile', 'description', 'requirements', 'benefits'])
    
    required_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits',
                    'len_description', 'len_requirements', 'len_benefits', 'len_company_profile',
                    'description_url_cnt', 'requirements_url_cnt', 'benefits_url_cnt', 'company_profile_url_cnt',
                    'description_email_cnt', 'requirements_email_cnt', 'benefits_email_cnt', 'company_profile_email_cnt',
                    'description_phone_cnt', 'requirements_phone_cnt', 'benefits_phone_cnt', 'company_profile_phone_cnt',
                    'salary_missing', 'salary_dash_cnt', 'telecommuting', 'has_company_logo', 'has_questions',
                    'employment_type', 'required_experience', 'required_education', 'industry', 'function', '__all_text__']
    final_input_df = input_df.reindex(columns=required_cols, fill_value=None)
    
    proba = model.predict_proba(final_input_df)[0]
    pred = np.argmax(proba)

    st.subheader("📊 Prediction Result")
    st.markdown(
        f"""
        **Prediction:** {"🟥 FAKE JOB" if pred == 1 else "🟩 REAL JOB"}  
        **Confidence:** {proba[pred]*100:.2f}%
        """
    )
    st.progress(int(proba[pred]*100))
    st.table(pd.DataFrame({"Class": ["Real (0)", "Fake (1)"], "Probability": [proba[0], proba[1]]}))

    if pred == 1:
        st.markdown("---")
        st.subheader("🕵️ Why this job is likely a scam:")
        reasons = explain_prediction(final_input_df)
        for reason in reasons:
            st.warning(f"- {reason}")

# ------------------------
# Model Metrics Section
# ------------------------
if show_metrics:
    st.markdown("---")
    st.subheader("📈 Model Performance (on validation set)")
    st.metric("ROC-AUC", "0.988")
    st.metric("Accuracy", "96.9%")

if show_confusion:
    st.markdown("### Confusion Matrix")
    cm = np.array([[3310, 93], [15, 158]])
    labels = ["Real (0)", "Fake (1)"]
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

# ------------------------
# Footer
# ------------------------
st.markdown("---")
st.caption("💡 Built with Streamlit | Logistic Regression | TF-IDF | Kaggle Fake Job Posting Dataset")