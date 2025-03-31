from warnings import catch_warnings

import streamlit as st
import pandas as pd
import numpy as np
import re
import gdown
import os
# import requests

from sklearn.base import BaseEstimator, ClassifierMixin
from yahooquery import Ticker
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import joblib
# from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier

# Function to interpret model results
def resultstring(arr):
    if arr[0] == 1:
        return "High Risk"
    elif arr[0] == 0:
        return "Low Risk"
    else:
        return "Error"

# Custom wrapper to use the neural network with sklearn's VotingClassifier
class KerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold

    def fit(self, X, y):
        return self  # No need to re-train the neural network

    def predict(self, X):
        X_nn = X.copy()
        X_nn[:] = X_nn[:].div(X_nn['X10'], axis=0)
        X_nn = X_nn.drop(columns=['X10'])
        return (self.model.predict(X_nn).flatten() >= self.threshold).astype(int)  # Ensure 1D

    def predict_proba(self, X):
        X_nn = X.copy()
        X_nn[:] = X_nn[:].div(X_nn['X10'], axis=0)
        X_nn = X_nn.drop(columns=['X10'])
        prob_1 = self.model.predict(X_nn).flatten()  # Make sure it's 1D
        prob_0 = 1 - prob_1  # Compute P(0)
        return np.vstack([prob_0, prob_1]).T  # Stack into (N, 2) shape

# Custom Random Forest Classifier with adjustable threshold
class RandomForestWithThreshold(BaseEstimator, ClassifierMixin):
    def __init__(self, rf_model, threshold=0.5):
        self.rf_model = rf_model  # The pre-trained random forest model
        self.threshold = threshold  # The custom threshold for classification

    def fit(self, X, y):
        self.rf_model.fit(X, y)
        return self

    def predict(self, X):
        probs = self.rf_model.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int).flatten()  # Ensure 1D

    def predict_proba(self, X):
        return np.asarray(self.rf_model.predict_proba(X))  # Ensure it's a NumPy array

# def download_from_google_drive(file_id, destination):
    # URL = f"https://drive.google.com/uc?export=download&id={file_id}"
    # session = requests.Session()
    # response = session.get(URL, stream=True)

    # with open(destination, "wb") as file:
        # for chunk in response.iter_content(chunk_size=128):
            # file.write(chunk)

# Cache data loading and preprocessing
@st.cache_resource
def load_models():
    # Construct the download URLs
    url_rf4 = "https://drive.google.com/uc?id=1aOJZZojkBHPKdVDr-sAHOtSfQ-2UUtly"  # Converted to direct link
    url_voting_clf2 = "https://drive.google.com/uc?id=1sWNi8FrnSI0w3f4MYwvhsCz_h1mpRGvY"  # Converted to direct link
    
    # Download the models
    # gdown.download(url_rf4, "rf4.pkl", quiet=False)
    # gdown.download(url_voting_clf2, "voting_clf2.pkl", quiet=False)
    
    # file_id_rf4 = "1aOJZZojkBHPKdVDr-sAHOtSfQ-2UUtly"  # Replace with your file's ID
    # file_id_voting_clf2 = "1sWNi8FrnSI0w3f4MYwvhsCz_h1mpRGvY"  # Replace with your file's ID

    # download_from_google_drive(file_id_rf4, "rf4.pkl")
    # download_from_google_drive(file_id_voting_clf2, "voting_clf2.pkl")

    def download_file(url, filename):
        if not os.path.exists(filename):
            os.system(f"wget --no-check-certificate '{url}' -O {filename}")

    # Download the files
    download_file(url_rf4, "rf4.pkl")
    download_file(url_voting_clf2, "voting_clf2.pkl")

    # Load the models after downloading
    nn = load_model('smote_nn1.keras')
    rf4 = joblib.load('rf4.pkl')
    voting_clf2 = joblib.load('voting_clf2.pkl')
    
    return rf4, nn, voting_clf2


rf4,nn,voting_clf2 = load_models()

# Handle session state to preserve values
if 'saved_text1' not in st.session_state:
    st.session_state.saved_text1 = None
if 'saved_text2' not in st.session_state:
    st.session_state.saved_text2 = None
if 'saved_market_value' not in st.session_state:
    st.session_state.saved_market_value = 0
if 'saved_year' not in st.session_state:
    st.session_state.saved_year = 2024

# Function to extract financial metrics for the latest year
def extract_latest_year_financial_metrics(text):
    metrics = {
        'X1': r'Current Assets(?:\n|\s)*([\d,]+)',
        'X2': r'Cost of Revenue(?:\n|\s)*([\d,]+)',  # Update to match the correct phrase if needed
        'X4': r'EBITDA(?:\n|\s)*([\d,]+)',  # Update to match the correct phrase if needed
        'X5': r'Inventory(?:\n|\s)*([\d,]+)',
        'X6': r'Net Income(?:\n|\s)*([\d,]+)',  # Update to match the correct phrase if needed
        'X7': r'Receivables(?:\n|\s)*([\d,]+)',
        'X8': r'Market value(?:\n|\s)*([\d,]+)',  # Update to match the correct phrase if needed
        'X9': r'Total Revenue(?:\n|\s)*([\d,]+)',
        'X10': r'Total Assets(?:\n|\s)*([\d,]+)',
        'X11': r'Long Term Debt(?:\n|\s)*([\d,]+)',
        'X12': r'EBIT(?:\n|\s)*([\d,]+)',  # Update to match the correct phrase if needed
        'X13': r'Gross Profit(?:\n|\s)*([\d,]+)',  # Update to match the correct phrase if needed
        'X14': r'Current Liabilities(?:\n|\s)*([\d,]+)',
        'X15': r'Retained Earnings(?:\n|\s)*([\d,]+)',
        # 'X16': r'Total Revenue(?:\n|\s)*([\d,]+)',
        'X17': r'Total Liabilities Net Minority Interest(?:\n|\s)*([\d,]+)',
        'X18': r'Operating Expense(?:\n|\s)*([\d,]+)',  # Update to match the correct phrase if needed
    }

    extracted_data = {}
    for key, pattern in metrics.items():
        match = re.search(pattern, text, re.IGNORECASE)
        extracted_data[key] = float(match.group(1).replace(',', '')) if match else 0

    # Create the DataFrame
    df = pd.DataFrame([{**extracted_data}])
    return df

def get_all_data(text, market_value):
    df = extract_latest_year_financial_metrics(text)
    df['X8'] = market_value
    # Calculate X3 as the sum of Accumulated Depreciation and Accumulated Amortization
    depreciation_match = re.search(r'Accumulated Depreciation(?:\n|\s)*-?([\d,]+)', text, re.IGNORECASE)
    amortization_match = re.search(r'Accumulated Amortization(?:\n|\s)*-?([\d,]+)', text, re.IGNORECASE)
    ordinary_shares_match = re.search(r'Ordinary Shares Number(?:\n|\s)*-?([\d,]+)', text, re.IGNORECASE)

    depreciation = abs(int(depreciation_match.group(1).replace(',', ''))) if depreciation_match else 0
    amortization = abs(int(amortization_match.group(1).replace(',', ''))) if amortization_match else 0
    shares = int(ordinary_shares_match.group(1).replace(',', '')) if ordinary_shares_match else 0
    df['X3'] = depreciation + amortization if depreciation or amortization else 0
    if shares:
        df['X8'] = df['X8'] * shares
    return df

st.title('Bankruptcy Calculator')
# Create an empty DataFrame to store x_data
if 'x_data' not in st.session_state:
    st.session_state.x_data = pd.DataFrame()

# Handle session state to preserve values
if 'saved_stock' not in st.session_state:
    st.session_state.saved_stock = None
if 'saved_text1' not in st.session_state:
    st.session_state.saved_text1 = None
if 'saved_text2' not in st.session_state:
    st.session_state.saved_text2 = None
if 'saved_market_value' not in st.session_state:
    st.session_state.saved_market_value = 0
#if 'saved_year' not in st.session_state:
    #st.session_state.saved_year = 2024

def calc():
    calculate = st.button(label="Calculate Bankruptcy Probability")
    if calculate:
        print(st.session_state.x_data.head())
        y_prob = voting_clf2.predict_proba(st.session_state.x_data)[:, 1]
        comb_threshold = 0.23284920588731767
        y_pred = (y_prob >= comb_threshold).astype(int)

        rf_prob = rf4.predict_proba(st.session_state.x_data)[:, 1]
        rf_threshold = 0.28
        rf_pred = (rf_prob >= rf_threshold).astype(int)

        X_nn = st.session_state.x_data.copy()
        X_nn[:] = X_nn[:].div(X_nn['X10'], axis=0)
        X_nn = X_nn.drop(columns=['X10'])
        nn_prob = nn.predict(X_nn)
        nn_threshold = 0.48695409297943115
        nn_pred = (nn_prob >= nn_threshold).astype(int)

        st.write("Random Forest Prediction: " + resultstring(rf_pred))
        st.write("Neural Network Prediction: " + resultstring(nn_pred))
        st.write("Combined Prediction: " + resultstring(y_pred))


# Streamlit UI
option = st.selectbox("Select input option:", ["Stock Symbol", "Annual Financial Statements", "Manually Enter Information"])
if option == "Stock Symbol":
    input_name = st.text_input(label="Enter Stock Symbol", value=st.session_state.saved_stock)
    st.session_state.saved_stock = input_name
    company = None
    if input_name:
        company = input_name.upper()
    if company:
        try:
            ticker = Ticker(company)
            financials = ticker.income_statement()
            position = ticker.balance_sheet()
            income = pd.DataFrame(financials).tail(1)
            balance = pd.DataFrame(position).tail(1)
            price = ticker.price
            st.write(income)
            st.write(balance)
            st.write(price)
            df = pd.DataFrame()
            # Define the columns and their corresponding sources
            column_mapping = {
                'X1': 'CurrentAssets',
                'X2': 'CostOfRevenue',
                'X4': 'EBITDA',
                'X5': 'Inventory',
                'X6': 'NetIncome',
                'X7': 'Receivables',
                'X9': 'TotalRevenue',
                'X10': 'TotalAssets',
                'X11': 'LongTermDebt',
                'X12': 'EBIT',
                'X13': 'GrossProfit',
                'X14': 'CurrentLiabilities',
                'X15': 'RetainedEarnings',
                # 'X16': 'TotalRevenue',  # This seems to be a duplicate of X9
                'X17': 'TotalLiabilitiesNetMinorityInterest',
                'X18': 'OperatingExpense'
            }

            # Assign values to the new DataFrame, checking if the column exists
            for col, source in column_mapping.items():
                if source in balance.columns or source in income.columns or source in price:
                    # Handle sources from different DataFrames or objects (balance, income, price)
                    if source in balance.columns:
                        df[col] = balance[source]
                    elif source in income.columns:
                        df[col] = income[source]
                else:
                    df[col] = 0  # If column is missing, assign NaN

            if "AccumulatedDepreciation" in balance.columns and "AccumulatedAmortization" in balance.columns:
                df["X3"] = abs(balance["AccumulatedDepreciation"] + balance["AccumulatedAmortization"])
            elif "AccumulatedDepreciation" in balance.columns:
                df["X3"] = abs(balance["AccumulatedDepreciation"])
            elif "AccumulatedAmortization" in balance.columns:
                df["X3"] = abs(balance["AccumulatedAmortization"])
            else:
                df['X3'] = 0
            df['X8'] = price.get(company, {}).get("regularMarketPrice", 0) * balance["OrdinarySharesNumber"] # Default to 0 if key is missing in price
            # x_data['year'] = balance["asOfDate"].dt.year
            cols = list(df.columns)[:-2]
            # order = ["year"] + cols[0:2] + ['X3'] + cols[2:6] + ['X8'] + cols[6:]
            order = cols[0:2] + ['X3'] + cols[2:6] + ['X8'] + cols[6:]
            st.session_state.x_data = df[order]
            # Always display the extracted data if available
            if st.session_state.x_data is not None:
                st.write(st.session_state.x_data)
            calc()
        except ValueError:
           st.write("Error! Stock not found!")

elif option == "Annual Financial Statements":
    try:
        # Textbox for income statement input
        text_input1 = st.text_area(label="Paste the income statement here:", value=st.session_state.saved_text1)

        # Textbox for balance sheet input
        text_input2 = st.text_area(label="Paste the balance sheet here:", value=st.session_state.saved_text2)

        # Textbox for market_value input
        market_value = st.number_input(label="Enter market value of stock:", value=st.session_state.saved_market_value)

        # Textbox for year_input
        # year = st.number_input(label="Enter year of financial statements:", value=st.session_state.saved_year)
        # st.session_state.saved_year = year

        clicked = st.button("Extract relevant financial data")

        if clicked and text_input1.strip() and text_input2.strip() :
            st.session_state.saved_text1 = text_input1
            st.session_state.saved_text2 = text_input2
            st.session_state.saved_market_value = market_value
            # Extract and display the financial metrics for the latest year
            text_input = text_input1 + text_input2
            df = get_all_data(text_input, market_value)
            # df['year'] = year
            cols = list(df.columns)[:-1]
            # order = ["year"] + cols[0:2] + ['X3'] + cols[2:]
            order = cols[0:2] + ['X3'] + cols[2:]
            df = df[order]
            st.session_state.x_data = df
        # Always display the extracted data if available
        if st.session_state.x_data is not None:
            st.write(st.session_state.x_data)
        calc()
    except ValueError:
        st.write("Error! Please try again!")

        # Option to download the DataFrame as CSV
        # csv = x_data.to_csv(index=False)
        # st.download_button(label="Download CSV", data=csv, file_name='financial_metrics_latest_year.csv', mime='text/csv')

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #663866, #446455); /* Dark blue to dark green */
        height: 100vh; /* Full height */
        padding: 20px; /* Optional padding */
        color: white; /* Change text color for better visibility */
    }
    </style>
    """,
    unsafe_allow_html=True
)
