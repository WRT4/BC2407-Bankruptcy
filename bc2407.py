import streamlit as st
import pandas as pd
import numpy as np
import re
from yahooquery import Ticker
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Cache data loading and preprocessing
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("american_bankruptcy.csv")
    y_train = df["status_label"]
    X_train = df.drop(columns=["company_name", "status_label", 'year'])
    # df_train2 = df[(df["year"]<2015)]
    # df_val = df[(df["year"]>=2015)]
    # X_train2 = df_train2.drop(columns=["company_name", "status_label", 'year'])
    # y_train2 = df_train2["status_label"]
    # X_val2 = df_val.drop(columns=["company_name", "status_label", 'year'])
    # y_val2 = df_val["status_label"]

    return X_train, y_train

@st.cache_resource
def train_models(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=100, random_state=99)
    rf.fit(X_train, y_train)
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    # y_val_encoded = le.transform(y_val2)

    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dropout(0.5, seed=99))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train_encoded, epochs=10, batch_size=32, verbose=1)

    return rf, model

X_train, y_train = load_and_preprocess_data()
rf, nn = train_models(X_train, y_train)

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
        'X16': r'Total Revenue(?:\n|\s)*([\d,]+)',
        'X17': r'Total Liabilities Net Minority Interest(?:\n|\s)*([\d,]+)',
        'X18': r'Operating Expense(?:\n|\s)*([\d,]+)',  # Update to match the correct phrase if needed
    }

    extracted_data = {}
    for key, pattern in metrics.items():
        match = re.search(pattern, text, re.IGNORECASE)
        extracted_data[key] = float(match.group(1).replace(',', '')) if match else None

    # Create the DataFrame
    df = pd.DataFrame([{**extracted_data}])

    return df

def get_all_data(text, market_value):
    df = extract_latest_year_financial_metrics(text)
    df['X8'] = market_value
    # Calculate X3 as the sum of Accumulated Depreciation and Accumulated Amortization
    depreciation_match = re.search(r'Accumulated Depreciation(?:\n|\s)*-?([\d,]+)', text, re.IGNORECASE)
    amortization_match = re.search(r'Accumulated Amortization(?:\n|\s)*-?([\d,]+)', text, re.IGNORECASE)

    depreciation = abs(int(depreciation_match.group(1).replace(',', ''))) if depreciation_match else 0
    amortization = abs(int(amortization_match.group(1).replace(',', ''))) if amortization_match else 0
    df['X3'] = depreciation + amortization if depreciation or amortization else None
    return df

st.title('Bankruptcy Calculator')
# Create an empty DataFrame
x_data = pd.DataFrame()

# Handle session state to preserve values
if 'saved_text1' not in st.session_state:
    st.session_state.saved_text1 = None
if 'saved_text2' not in st.session_state:
    st.session_state.saved_text2 = None
if 'saved_market_value' not in st.session_state:
    st.session_state.saved_market_value = 0
#if 'saved_year' not in st.session_state:
    #st.session_state.saved_year = 2024

# Streamlit UI
option = st.selectbox("Select input option:", ["Stock Name", "Annual Financial Statements", "Manually Enter Information"])
if option == "Stock Name":
    company = st.text_input(label="Enter stock name").upper()
    if company:
        ticker = Ticker(company)
        financials = ticker.income_statement()
        position = ticker.balance_sheet()
        income = pd.DataFrame(financials).tail(1)
        balance = pd.DataFrame(position).tail(1)
        price = ticker.price
        st.write(income)
        st.write(balance)
        st.write(price)
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
            'X16': 'TotalRevenue',  # This seems to be a duplicate of X9
            'X17': 'TotalLiabilitiesNetMinorityInterest',
            'X18': 'OperatingExpense'
        }

        # Assign values to the new DataFrame, checking if the column exists
        for col, source in column_mapping.items():
            if source in balance.columns or source in income.columns or source in price:
                # Handle sources from different DataFrames or objects (balance, income, price)
                if source in balance.columns:
                    x_data[col] = balance[source]
                elif source in income.columns:
                    x_data[col] = income[source]
            else:
                x_data[col] = np.nan  # If column is missing, assign NaN

        if "AccumulatedDepreciation" in balance.columns and "AccumulatedAmortization" in balance.columns:
            x_data["X3"] = abs(balance["AccumulatedDepreciation"] + balance["AccumulatedAmortization"])
        elif "AccumulatedDepreciation" in balance.columns:
            x_data["X3"] = abs(balance["AccumulatedDepreciation"])
        elif "AccumulatedAmortization" in balance.columns:
            x_data["X3"] = abs(balance["AccumulatedAmortization"])
        else:
            x_data['X3'] = np.nan
        x_data['X8'] = price.get(company, {}).get("regularMarketPrice", np.nan)  # Default to NaN if key is missing in price
        # x_data['year'] = balance["asOfDate"].dt.year
        cols = list(x_data.columns)[:-2]
        # order = ["year"] + cols[0:2] + ['X3'] + cols[2:6] + ['X8'] + cols[6:]
        order = cols[0:2] + ['X3'] + cols[2:6] + ['X8'] + cols[6:]
        x_data = x_data[order]
        st.write(x_data)

elif option == "Annual Financial Statements":

    # Textbox for income statement input
    text_input1 = st.text_area(label="Paste the income statement here:", value=st.session_state.saved_text1)
    st.session_state.saved_text1 = text_input1

    # Textbox for balance sheet input
    text_input2 = st.text_area(label="Paste the balance sheet here:", value=st.session_state.saved_text2)
    st.session_state.saved_text2 = text_input2

    # Textbox for market_value input
    market_value = st.number_input(label="Enter market value of stock:", value=st.session_state.saved_market_value)
    st.session_state.saved_market_value = market_value

    # Textbox for year_input
    # year = st.number_input(label="Enter year of financial statements:", value=st.session_state.saved_year)
    # st.session_state.saved_year = year

    clicked = st.button("Extract relevant financial data")

    if clicked and text_input1.strip() and text_input2.strip() :
        # Extract and display the financial metrics for the latest year
        text_input = text_input1 + text_input2
        x_data = get_all_data(text_input, market_value)
        # x_data['year'] = year
        cols = list(x_data.columns)[:-1]
        # order = ["year"] + cols[0:2] + ['X3'] + cols[2:]
        order = cols[0:2] + ['X3'] + cols[2:]
        x_data = x_data[order]
        st.write(x_data)

        # Option to download the DataFrame as CSV
        # csv = x_data.to_csv(index=False)
        # st.download_button(label="Download CSV", data=csv, file_name='financial_metrics_latest_year.csv', mime='text/csv')

calculate = st.button(label="Calculate Bankruptcy Probability")
if calculate:

    rf_pred = rf.predict(x_data)
    rf_pred_prob = rf.predict_proba(x_data)

    nn_pred = nn.predict(x_data)

    st.write("Random Forest Prediction: " + str(rf_pred) + str(rf_pred_prob))
    st.write("Neural Network Prediction: " + str(nn_pred))
