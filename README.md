# FAKE-JOB-DETECTOR
# 🕵️ Fake Job Posting Detector

An AI-powered web application that helps users detect fraudulent job postings by analyzing their content for common red flags. This tool uses a pre-trained machine learning model to provide a real-time prediction and explanation.

## 🚀 Key Features

* **Real-Time Prediction:** Instantly analyzes a job posting and predicts if it's real or fake.
* **Explainable AI:** Provides a clear, human-readable list of reasons why a posting might be a scam.
* **Web Scraping:** Automatically fetches job posting details from a given LinkedIn URL.
* **User-Friendly Interface:** Built with Streamlit, providing an interactive and intuitive experience.
* **Customizable Theme:** Features a custom, light blue color scheme for a comfortable viewing experience.

## 🧠 Model and Methodology

The core of this application is a machine learning model trained on a dataset of real and fake job postings.

* **Model:** We use a **Logistic Regression** classifier for its interpretability and efficiency in handling binary classification tasks.
* **Feature Engineering:** The model's performance relies on robust feature engineering. The text from the job posting is converted into numerical features using a **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorizer. Additional handcrafted features, such as the number of URLs, emails, and phone numbers in the posting, are also used to improve accuracy.

## 💻 How to Run the App Locally

Follow these simple steps to get the Fake Job Detector running on your machine.

### Prerequisites

* Python 3.7 or higher installed.

### Step-by-Step Guide

1.  **Clone or Download the Project:**
     clone the repository b.

2.  **Set Up the Environment:**
    Open your terminal or command prompt, navigate to the project directory, and create a virtual environment to manage dependencies:
    ```bash
    python -m venv venv
    ```
    Activate the virtual environment:
    * **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
    * **On Windows:**
        ```bash
        venv\Scripts\activate
        ```

3.  **Install Dependencies:**
    Install all required libraries using pip:
    ```bash
    pip install pandas numpy scikit-learn joblib requests beautifulsoup4 streamlit matplotlib seaborn
    ```

4.  **Add the Trained Model:**
    The application requires a pre-trained model file. Download the `fake_job_detector.joblib` file and place it in the same directory as `streamlit_app.py`.
    or you can run python main.py on the data set fake_job_posting.csv

6.  **Run the Application:**
    With your virtual environment activated, run the Streamlit app with the following command:
    ```bash
    streamlit run streamlit_app.py
    ```

    The application will automatically open in your default web browser.

## 📄 File Structure

* `streamlit_app.py`: The main Python script for the Streamlit web application.
* `fake_job_detector.joblib`: The pre-trained machine learning model file.
* `README.md`: This file.

## 🙏 Acknowledgements

* **Dataset:** The model was trained on the [Kaggle Fake Job Posting Prediction dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction/data).
* **Libraries:** Built with the incredible open-source tools: Streamlit, scikit-learn, requests, and BeautifulSoup.
