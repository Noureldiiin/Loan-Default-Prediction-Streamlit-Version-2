import streamlit as st
import pandas as pd
import numpy as np
import pickle
import subprocess
import sys

st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="🏦",
)

subprocess.check_call([sys.executable, "-m", "pip", "install", 'imblearn'])
subprocess.check_call([sys.executable, "-m", "pip", "install", 'xgboost'])

# Load your model and input feature names here
modelname = 'finalized_model.sav'
inputname = 'input_name.sav'
model = pickle.load(open(modelname, 'rb'))
inputs = pickle.load(open(inputname, 'rb'))

def predict_loan_default(Age, Income, LoanAmount, CreditScore, MonthsEmployed, NumCreditLines, InterestRate, LoanTerm, DTIRatio, Education, EmploymentType, MaritalStatus, HasMortgage, HasDependents, LoanPurpose, HasCoSigner):
    # Create a DataFrame with input features
    input_data = pd.DataFrame(columns=inputs)

    # Define a list of input feature names for progress tracking
    input_features = [
        "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed", 
        "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio", 
        "Education", "EmploymentType", "MaritalStatus", "HasMortgage", 
        "HasDependents", "LoanPurpose", "HasCoSigner"
    ]

    # Create a progress bar
    progress_bar = st.progress(0)

    # Function to update the progress bar
    def update_progress(progress):
        progress_bar.progress(progress / len(input_features))

    # Iterate over input features and update the progress bar
    for idx, feature in enumerate(input_features):
        # Update the progress bar with the current feature's index
        update_progress(idx)

        # Read the feature values and make predictions here

    # Make a prediction using your model
    prediction = model.predict(input_data)[0]
    return prediction

def main():
    st.title("Loan Default Prediction")
    custom_options = {
        "Education": ["Bachelor's", "Master's", "High School", "Other"],
        "EmploymentType": ["Full-time", "Unemployed", "Part-time", "Self-employed"],
        "MaritalStatus": ["Divorced", "Married", "Single"],
        "HasMortgage": ["Yes", "No"],
        "HasDependents": ["Yes", "No"],
        "LoanPurpose": ["Auto", "Business", "Home", "Education", "Other"],
        "HasCoSigner": ["Yes", "No"]
    }

    # Create input fields for each column
    input_data = {}

    if "Age" in inputs:
        Age = st.number_input("Age", min_value=0)
        input_data['Age'] = Age
    if "Income" in inputs:
        Income = st.number_input("Income", min_value=0)
        input_data['Income'] = Income
    if "LoanAmount" in inputs:
        LoanAmount = st.number_input("LoanAmount", min_value=0)
        input_data['LoanAmount'] = LoanAmount
    if "CreditScore" in inputs:
        CreditScore = st.number_input("CreditScore", min_value=0)
        input_data['CreditScore'] = CreditScore
    if "MonthsEmployed" in inputs:
        MonthsEmployed = st.number_input("MonthsEmployed", min_value=0)
        input_data['MonthsEmployed'] = MonthsEmployed
    if "NumCreditLines" in inputs:
        NumCreditLines = st.number_input("NumCreditLines", min_value=0)
        input_data['NumCreditLines'] = NumCreditLines
    if "LoanTerm" in inputs:
        LoanTerm = st.number_input("LoanTerm", min_value=0)
        input_data['LoanTerm'] = LoanTerm
    if "InterestRate" in inputs:
        InterestRate = st.number_input("InterestRate", min_value=0.0, step=0.01)
        input_data['InterestRate'] = InterestRate
    if "DTIRatio" in inputs:
        DTIRatio = st.number_input("DTIRatio", min_value=0.0, step=0.01)
        input_data['DTIRatio'] = DTIRatio

    if "Education" in inputs:
        Education = st.selectbox("Education", custom_options["Education"])
        input_data['Education'] = Education
    if "EmploymentType" in inputs:
        EmploymentType = st.selectbox("EmploymentType", custom_options["EmploymentType"])
        input_data['EmploymentType'] = EmploymentType
    if "MaritalStatus" in inputs:
        MaritalStatus = st.selectbox("MaritalStatus", custom_options["MaritalStatus"])
        input_data['MaritalStatus'] = MaritalStatus
    if "HasMortgage" in inputs:
        HasMortgage = st.selectbox("HasMortgage", custom_options["HasMortgage"])
        input_data['HasMortgage'] = HasMortgage
    if "HasDependents" in inputs:
        HasDependents = st.selectbox("HasDependents", custom_options["HasDependents"])
        input_data['HasDependents'] = HasDependents
    if "LoanPurpose" in inputs:
        LoanPurpose = st.selectbox("LoanPurpose", custom_options["LoanPurpose"])
        input_data['LoanPurpose'] = LoanPurpose
    if "HasCoSigner" in inputs:
        HasCoSigner = st.selectbox("HasCoSigner", custom_options["HasCoSigner"])
        input_data['HasCoSigner'] = HasCoSigner

    if st.button("Predict"):
        # Make the prediction
        prediction = predict_loan_default(Age, Income, LoanAmount, CreditScore, MonthsEmployed, NumCreditLines, InterestRate, LoanTerm, DTIRatio, Education, EmploymentType, MaritalStatus, HasMortgage, HasDependents, LoanPurpose, HasCoSigner)

        # Display the prediction
        if prediction == 1:
            st.write("Loan Default: Yes")
        else:
            st.write("Loan Default: No")
    
    
main()
