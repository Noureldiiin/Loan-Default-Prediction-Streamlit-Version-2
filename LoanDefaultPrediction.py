import streamlit as st
import pandas as pd
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer , KNNImputer
from sklearn.preprocessing import StandardScaler , MinMaxScaler , RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.naive_bayes import MultinomialNB , GaussianNB
from sklearn.model_selection import train_test_split , cross_validate
from sklearn.metrics import accuracy_score , recall_score , precision_score , f1_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler , RobustScaler , MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from datasist.structdata import detect_outliers
from sklearn.metrics import silhouette_score
import subprocess
import sys

st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="🏦",
)

subprocess.check_call([sys.executable, "-m", "pip", "install", 'imblearn'])
subprocess.check_call([sys.executable, "-m", "pip", "install", 'xgboost'])
# pip install imblearn
modelname = 'finalized_model.sav'
inputname = 'input_name.sav'
model = pickle.load(open(modelname, 'rb'))
inputs = pickle.load(open(inputname, 'rb'))
# model = joblib.load("Model.pkl")
# inputs = joblib.load("inputs.pkl")

def predict_loan_default(Age, Income, LoanAmount, CreditScore, MonthsEmployed, NumCreditLines, InterestRate, LoanTerm, DTIRatio, Education, EmploymentType, MaritalStatus, HasMortgage, HasDependents, LoanPurpose, HasCoSigner):
    # Create a DataFrame with input features
    input_data = pd.DataFrame(columns=inputs)
    
    input_data.at[0, 'Age'] = Age
    input_data.at[0, 'Income'] = Income
    input_data.at[0, 'LoanAmount'] = LoanAmount
    input_data.at[0, 'CreditScore'] = CreditScore
    input_data.at[0, 'MonthsEmployed'] = MonthsEmployed
    input_data.at[0, 'NumCreditLines'] = NumCreditLines
    input_data.at[0, 'InterestRate'] = InterestRate
    input_data.at[0, 'LoanTerm'] = LoanTerm
    input_data.at[0, 'DTIRatio'] = DTIRatio
    input_data.at[0, 'Education'] = Education
    input_data.at[0, 'EmploymentType'] = EmploymentType
    input_data.at[0, 'MaritalStatus'] = MaritalStatus
    input_data.at[0, 'HasMortgage'] = HasMortgage
    input_data.at[0, 'HasDependents'] = HasDependents
    input_data.at[0, 'LoanPurpose'] = LoanPurpose
    input_data.at[0, 'HasCoSigner'] = HasCoSigner
    
    bins = [0, 25, 35, 45, 55, 65, np.inf]
    labels = ['<25', '25-34', '35-44', '45-54', '55-64', '65+']
    input_data['AgeCategory'] = pd.cut(input_data['Age'], bins=bins, labels=labels)

    # Calculate Income to Debt Ratio
    input_data['IncomeDebtRatio'] = input_data['Income'] / input_data['DTIRatio']

    # Calculate Loan Amount to Income Ratio
    input_data['LoanIncomeRatio'] = input_data['LoanAmount'] / input_data['Income']

    # Create Credit Score categories
    input_data['CreditScoreCategory'] = pd.cut(input_data['CreditScore'], bins=[0, 600, 700, 800, 850], labels=['Poor', 'Fair', 'Good', 'Excellent'])

    # Create Employment Length categories
    input_data['EmploymentLengthCategory'] = pd.cut(input_data['MonthsEmployed'], bins=[0, 12, 36, 60, np.inf], labels=['<1 year', '1-3 years', '3-5 years', '5+ years'])

    # Calculate Credit Utilization
    input_data['CreditUtilization'] = input_data['NumCreditLines'] / input_data['CreditScore']

    # Create Interest Rate categories
    input_data['InterestRateCategory'] = pd.cut(input_data['InterestRate'], bins=[0, 5, 10, 15, 20, np.inf], labels=['<5%', '5-10%', '10-15%', '15-20%', '20+%'])

    # Create Loan Term categories
    input_data['LoanTermCategory'] = pd.cut(input_data['LoanTerm'], bins=[0, 12, 24, 36, 48, 60, np.inf], labels=['1 year', '2 years', '3 years', '4 years', '5 years', '5+ years'])

    # Drop original columns used for feature engineering
    columns_to_drop = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'NumCreditLines', 'DTIRatio', 'InterestRate', 'LoanTerm']
    input_data.drop(columns=columns_to_drop, inplace=True)
    
    
    # Make a prediction
    prediction = model.predict(input_data)[0]
    return prediction

def main():
    st.title("Loan Default Prediction")
    custom_options = {
        "Education": ["Bachelor's", "Master's", "High School", "Other"],  # Define custom options for Education
        "EmploymentType": ["Full-time", "Unemployed", "Part-time", "Self-employed"],  # Define custom options for EmploymentType
        "MaritalStatus": ["Divorced", "Married", "Single"],  # Define custom options for MaritalStatus
        "HasMortgage": ["Yes", "No"],  # Define custom options for HasMortgage
        "HasDependents": ["Yes", "No"],  # Define custom options for HasDependents
        "LoanPurpose": ["Auto", "Business", "Home", "Education", "Other"],  # Define custom options for LoanPurpose
        "HasCoSigner": ["Yes", "No"]  # Define custom options for HasCoSigner
    }
    # Create input fields for each column
    input_data = {}
    print(inputs)
    if "Age" in inputs:
        Age = st.number_input("Age", min_value=0)
    if "Income" in inputs:
        Income = st.number_input("Income", min_value=0)
    if "LoanAmount" in inputs:
        LoanAmount = st.number_input("LoanAmount", min_value=0)
    if "CreditScore" in inputs:
        CreditScore = st.number_input("CreditScore", min_value=0)
    if "MonthsEmployed" in inputs:
        MonthsEmployed = st.number_input("MonthsEmployed", min_value=0)
    if "NumCreditLines" in inputs:
        NumCreditLines = st.number_input("NumCreditLines", min_value=0)
    if "LoanTerm" in inputs:
        LoanTerm = st.number_input("LoanTerm", min_value=0)
    if "InterestRate" in inputs:
        InterestRate = st.number_input("InterestRate", min_value=0.0, step=0.01)
    if "DTIRatio" in inputs:
        DTIRatio = st.number_input("DTIRatio", min_value=0.0, step=0.01)

    if "Education" in inputs:
        Education = st.selectbox("Education", custom_options["Education"])
    if "EmploymentType" in inputs:
        EmploymentType = st.selectbox("EmploymentType", custom_options["EmploymentType"])
    if "MaritalStatus" in inputs:
        MaritalStatus = st.selectbox("MaritalStatus", custom_options["MaritalStatus"])
    if "HasMortgage" in inputs:
        HasMortgage = st.selectbox("HasMortgage", custom_options["HasMortgage"])
    if "HasDependents" in inputs:
        HasDependents = st.selectbox("HasDependents", custom_options["HasDependents"])
    if "LoanPurpose" in inputs:
        LoanPurpose = st.selectbox("LoanPurpose", custom_options["LoanPurpose"])
    if "HasCoSigner" in inputs:
        HasCoSigner = st.selectbox("HasCoSigner", custom_options["HasCoSigner"])


    if st.button("Predict"):
        # Make the prediction
        prediction = predict_loan_default(Age, Income, LoanAmount, CreditScore, MonthsEmployed, NumCreditLines, InterestRate, LoanTerm, DTIRatio, Education, EmploymentType, MaritalStatus, HasMortgage, HasDependents, LoanPurpose, HasCoSigner)
        
        print('ssssss')
        print(prediction)
        # Display the prediction
         if prediction == 1:
            st.button("Loan Default: Yes", key="popup_yes")
            if st.session_state.popup_yes:
                st.info("Loan Default: Yes")
        else:
            st.button("Loan Default: No", key="popup_no")
            if st.session_state.popup_no:
                st.info("Loan Default: No")
    
    
main()
