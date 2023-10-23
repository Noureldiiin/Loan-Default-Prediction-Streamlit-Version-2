import streamlit as st
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import locale
import time

# Load your data (replace 'df' with the actual DataFrame)
dfname = 'dataframe.sav'
df = pickle.load(open(dfname, 'rb'))

# Set Streamlit page config
st.set_page_config(page_title="Dashboard", page_icon="📈")

# Define the Streamlit app
st.title("Dashboard")

# Progress Bar
progress_bar = st.progress(0)

# Function to update progress bar
def update_progress_bar(progress, max_progress):
    progress_bar.progress(progress / max_progress)




# Display the range of income\f

min_income = df['Income'].min()
max_income = df['Income'].max()
st.subheader("Income Range")
st.write(f"The minimum income is: ${min_income:,.0f}")
st.write(f"The maximum income is: ${max_income:,.0f}")
update_progress_bar(1, 16)  # Progress: 1/16

# Display the most common loan purpose
most_common_loan_purpose = df['LoanPurpose'].mode().values[0]
st.subheader("Most Common Loan Purpose")
st.write(f"The most common loan purpose among borrowers is: {most_common_loan_purpose}")
update_progress_bar(2, 16)  # Progress: 2/16

# Calculate the overall default rate
total_records = len(df)
default_records = df['Default'].sum()
default_rate = default_records / total_records
st.subheader("Default Rate")
st.write(f"The overall default rate in the dataset is: {default_rate * 100:.1f}%")
update_progress_bar(3, 16)  # Progress: 3/16

# Calculate the average age of borrowers
average_age = df['Age'].mean()
st.subheader("Average Age")
st.write(f"The average age of borrowers is: {average_age:.2f} years")
update_progress_bar(4, 16)  # Progress: 4/16

# Default Rates by Education Level using Seaborn
education_default_rates = df.groupby('Education')['Default'].mean().reset_index()
plt.figure(figsize=(8, 6))
sns.barplot(x='Education', y='Default', data=education_default_rates)
plt.xlabel('Education Level')
plt.ylabel('Default Rate')
plt.title('Default Rates by Education Level')

st.subheader("Default Rates by Education Level")
st.pyplot(plt)
update_progress_bar(5, 16)  # Progress: 5/16

# Default Rates by Employment Type using Seaborn
employment_default_rates = df.groupby('EmploymentType')['Default'].mean().reset_index()
plt.figure(figsize=(8, 6))
sns.barplot(x='EmploymentType', y='Default', data=employment_default_rates)
plt.xlabel('Employment Type')
plt.ylabel('Default Rate')
plt.title('Default Rates by Employment Type')

st.subheader("Default Rates by Employment Type")
st.pyplot(plt.gcf())
update_progress_bar(6, 16)  # Progress: 6/16

# Default Rates for Unemployed Individuals by Education Level using Seaborn
employment_education_rates = df[df['EmploymentType'] == 'Unemployed'].groupby('Education')['Default'].mean().reset_index()
plt.figure(figsize=(8, 6))
sns.barplot(x='Education', y='Default', data=employment_education_rates)
plt.xlabel('Education')
plt.ylabel('Default Rate')
plt.title('Common Education level for Unemployed Individuals')

st.subheader("Common Education level for for Unemployed Individuals")
st.pyplot(plt.gcf())
update_progress_bar(7, 16)  # Progress: 7/16

# Default Rates by Loan Purpose for Unemployed Individuals using Seaborn
employment_purpose_rates = df[df['EmploymentType'] == 'Unemployed'].groupby('LoanPurpose')['Default'].sum().reset_index()
plt.figure(figsize=(8, 6))
sns.barplot(x='LoanPurpose', y='Default', data=employment_purpose_rates)
plt.xlabel('Loan Purpose')
plt.ylabel('Default Rate')
plt.title('Common Loan Purpose for Unemployed Individuals')

st.subheader("Common Loan Purpose for Unemployed Individuals")
st.pyplot(plt.gcf())
update_progress_bar(8, 16)  # Progress: 8/16

# Common Loan Purpose for Unemployed Individuals When Loan Purpose Equals Home
employment_purpose_rates_home = df[(df['EmploymentType'] == 'Unemployed') & (df['LoanPurpose'] == 'Home')].groupby('Education')['Default'].sum().reset_index()
plt.figure(figsize=(8, 6))
plt.pie(employment_purpose_rates_home['Default'], labels=employment_purpose_rates_home['Education'], autopct='%1.1f%%')
plt.title('Common Loan Purpose for Unemployed Individuals When Loan Purpose Equals Home')
st.subheader('Common Loan Purpose for Unemployed Individuals When Loan Purpose Equals Home')
st.pyplot(plt.gcf())


# Common Loan Purpose for Unemployed Individuals When Loan Purpose Equals Business
employment_purpose_rates_business = df[(df['EmploymentType'] == 'Unemployed') & (df['LoanPurpose'] == 'Business')].groupby('Education')['Default'].sum().reset_index()
plt.figure(figsize=(8, 6))
plt.pie(employment_purpose_rates_business['Default'], labels=employment_purpose_rates_business['Education'], autopct='%1.1f%%')
plt.title('Common Loan Purpose for Unemployed Individuals When Loan Purpose Equals Business')
st.subheader('Common Loan Purpose for Unemployed Individuals When Loan Purpose Equals Business')
st.pyplot(plt.gcf())

# Education When Co Signer Equals Yes
employment_purpose_rates_home = df[(df['HasCoSigner'] == 'Yes')].groupby('Education')['Default'].sum().reset_index()
plt.figure(figsize=(8, 6))
plt.pie(employment_purpose_rates_home['Default'], labels=employment_purpose_rates_home['Education'], autopct='%1.1f%%')
# plt.title('Education When Co Signer Equals Yes')
st.subheader('Education When Co Signer Equals Yes')
st.pyplot(plt.gcf())

# Education When Has Dependents Equals Yes
employment_purpose_rates_home = df[(df['HasDependents'] == 'Yes')].groupby('Education')['Default'].sum().reset_index()
plt.figure(figsize=(8, 6))
plt.pie(employment_purpose_rates_home['Default'], labels=employment_purpose_rates_home['Education'], autopct='%1.1f%%')
# plt.title('Education When Co Signer Equals Yes')
st.subheader('Education When Has Dependents Equals Yes')
st.pyplot(plt.gcf())


# Average Loan Term For Each Employment Type
employment_purpose_rates_home = df.groupby('EmploymentType')['LoanTerm'].mean().reset_index()
plt.figure(figsize=(8, 6))
plt.pie(employment_purpose_rates_home['LoanTerm'], labels=employment_purpose_rates_home['EmploymentType'], autopct='%1.1f%%')
# plt.title('Education When Co Signer Equals Yes')
st.subheader('Average Loan Term For Each Employment Type')
st.pyplot(plt.gcf())



# Average Interest Rate For Each Employment Type
employment_purpose_rates_home = df.groupby('EmploymentType')['InterestRate'].mean().reset_index()
plt.figure(figsize=(8, 6))
plt.pie(employment_purpose_rates_home['InterestRate'], labels=employment_purpose_rates_home['EmploymentType'], autopct='%1.1f%%')
# plt.title('Education When Co Signer Equals Yes')
st.subheader('Average Interest Rate For Each Employment Type')
st.pyplot(plt.gcf())


# Average Credit Score For Each Employment Type
employment_purpose_rates_home = df.groupby('EmploymentType')['CreditScore'].mean().reset_index()
plt.figure(figsize=(8, 6))
plt.pie(employment_purpose_rates_home['CreditScore'], labels=employment_purpose_rates_home['EmploymentType'], autopct='%1.1f%%')
# plt.title('Education When Co Signer Equals Yes')
st.subheader('Average Credit Score For Each Employment Type')
st.pyplot(plt.gcf())


# Calculate the average loan term (duration)
average_loan_term = df['LoanTerm'].mean()

# Display the average loan term in the Streamlit app
st.subheader("Average Loan Term")
st.write(f"The average loan term in the dataset is: {average_loan_term:.2f} months")
update_progress_bar(9, 16)  # Progress: 9/16

# Loan Defaults by Age using Seaborn
marital_status_default_rates = df.groupby('Age')['Default'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=marital_status_default_rates, x='Age', y='Default', color='blue')
plt.title('Loan Defaults by Age')
plt.xlabel('Age Category')
plt.ylabel('Default Rate')
plt.xticks(rotation=45)
plt.tight_layout()
st.subheader("Loan Defaults by Age")
st.pyplot(plt.gcf())
update_progress_bar(10, 16)  # Progress: 10/16

# Define age categories
age_categories = [
    "Young",
    "Middle-aged",
    "Senior"
]

# Function to categorize age
def create_small_age_category(age):
    if age < 35:
        return age_categories[0]
    elif age < 55:
        return age_categories[1]
    else:
        return age_categories[2]

# Add 'age_category' column to DataFrame
df['age_category'] = df['Age'].apply(create_small_age_category)

# Set the locale for formatting percentages
locale.setlocale(locale.LC_ALL, '')

# Calculate default rates by marital status
marital_status_default_rates = df.groupby('MaritalStatus')['Default'].mean().reset_index()

# Format default rates as percentages
marital_status_default_rates['FormattedDefaultRate'] = marital_status_default_rates['Default'].apply(
    lambda x: locale.format_string('%.2f%%', x * 100, grouping=True)
)

plt.figure(figsize=(10, 6))
sns.barplot(data=marital_status_default_rates, x='MaritalStatus', y='Default', color='blue')
plt.title('Loan Defaults by Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Default Rate')
plt.xticks(rotation=45)
plt.tight_layout()

st.subheader("Loan Defaults by Marital Status")
st.pyplot(plt.gcf())
update_progress_bar(11, 16)  # Progress: 11/16

# Set the locale for formatting percentages
locale.setlocale(locale.LC_ALL, '')

# Calculate default rates by the presence of dependents
dependents_default_rates = df.groupby('HasDependents')['Default'].mean().reset_index()

# Format default rates as percentages
dependents_default_rates['FormattedDefaultRate'] = dependents_default_rates['Default'].apply(
    lambda x: locale.format_string('%.2f%%', x * 100, grouping=True)
)

plt.figure(figsize=(8, 6))
sns.barplot(data=dependents_default_rates, x='HasDependents', y='Default', color='blue')
plt.title('Loan Defaults by Dependents')
plt.xlabel('Dependents')
plt.ylabel('Default Rate')
plt.tight_layout()

st.subheader("Loan Defaults by Dependents")
st.pyplot(plt.gcf())
update_progress_bar(12, 16)  # Progress: 12/16

# Set the locale for formatting percentages
locale.setlocale(locale.LC_ALL, '')

# Calculate average interest rates by employment type
employment_type_interest_rates = df.groupby('EmploymentType')['InterestRate'].mean().reset_index()

# Format interest rates as percentages
employment_type_interest_rates['FormattedInterestRate'] = employment_type_interest_rates['InterestRate'].apply(
    lambda x: locale.format_string('%.2f%%', x * 100, grouping=True)
)

plt.figure(figsize=(8, 6))
sns.barplot(data=employment_type_interest_rates, x='EmploymentType', y='InterestRate', color='blue')
plt.title('Interest Rate by Employment Type')
plt.xlabel('Employment Type')
plt.ylabel('Interest Rate')
plt.tight_layout()

st.subheader("Interest Rate by Employment Type")
st.pyplot(plt.gcf())
update_progress_bar(13, 16)  # Progress: 13/16

# Calculate the correlation coefficient between 'MonthsEmployed' and 'CreditScore'
selected_columns = ['MonthsEmployed', 'CreditScore']
correlation_coefficient = df[selected_columns].corr().iloc[0, 1]

# Display the correlation coefficient
st.subheader("Correlation between MonthsEmployed and CreditScore")
st.write(f"The correlation coefficient between 'MonthsEmployed' and 'CreditScore' is: {correlation_coefficient:.2f}")
update_progress_bar(14, 16)  # Progress: 14/16

# Set the locale for formatting percentages
locale.setlocale(locale.LC_ALL, '')

# Calculate default rates by the presence of a co-signer
cosigner_default_rates = df.groupby('HasCoSigner')['Default'].mean().reset_index()

# Format default rates as percentages
cosigner_default_rates['FormattedDefaultRate'] = cosigner_default_rates['Default'].apply(
    lambda x: locale.format_string('%.2f%%', x * 100, grouping=True)
)

plt.figure(figsize=(8, 6))
sns.barplot(data=cosigner_default_rates, x='HasCoSigner', y='Default', color='blue')
plt.title('Loan Defaults by Co-Signer')
plt.xlabel('Co-Signer')
plt.ylabel('Default Rate')
plt.tight_layout()

st.subheader("Loan Defaults by Co-Signer")
st.pyplot(plt.gcf())
update_progress_bar(15, 16)  # Progress: 15/16

# Set the locale for formatting counts
locale.setlocale(locale.LC_ALL, '')

# Calculate and format the counts of loan purposes
loan_purpose_counts = df['LoanPurpose'].value_counts().reset_index()
loan_purpose_counts.columns = ['LoanPurpose', 'Count']
loan_purpose_counts['FormattedCount'] = loan_purpose_counts['Count'].apply(lambda x: locale.format_string('%d', x, grouping=True))

plt.figure(figsize=(12, 6))
sns.barplot(data=loan_purpose_counts, x='LoanPurpose', y='Count', color='blue')
plt.title('Most Common Loan Purposes')
plt.xlabel('Loan Purpose')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()

st.subheader("Most Common Loan Purposes")
st.pyplot(plt.gcf())
update_progress_bar(16, 16)  # Progress: 16/16

# Calculate the correlation coefficient between 'Default Rate' and 'CreditScore'
selected_columns = ['Default', 'CreditScore']
correlation_coefficient = df[selected_columns].corr().iloc[0, 1]

st.subheader("Correlation between Default Rate and Credit Score")
st.write(f"The correlation coefficient between 'Default Rate' and 'CreditScore' is: {correlation_coefficient:.2f}")

# Calculate the correlation coefficient between 'Loan Amount' and 'CreditScore'
selected_columns = ['LoanAmount', 'CreditScore']
correlation_coefficient = df[selected_columns].corr().iloc[0, 1]

st.subheader("Correlation between Loan Amount and Credit Score")
st.write(f"The correlation coefficient between 'Loan Amount' and 'CreditScore' is: {correlation_coefficient:.2f}")

plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Education', y='Income', color='blue')
plt.title('Income Rates by Education Level')
plt.xlabel('Education Level')
plt.ylabel('Income Rate')
plt.xticks(rotation=45)
plt.tight_layout()

st.subheader("Income Rates by Education Level")
st.pyplot(plt.gcf())

plt.figure(figsize=(10, 6))
employment_type_default_rates = df.groupby('EmploymentType')['Default'].mean().reset_index()
sns.barplot(data=employment_type_default_rates, x='EmploymentType', y='Default', color='blue')
plt.title('Default Rates by Employment Type')
plt.xlabel('Employment Type')
plt.ylabel('Default Rate')
plt.xticks(rotation=45)
plt.tight_layout()

st.subheader("Default Rates by Employment Type")
st.pyplot(plt.gcf())

plt.figure(figsize=(8, 6))
mortgage_default_rates = df.groupby('HasMortgage')['Default'].mean().reset_index()
sns.barplot(data=mortgage_default_rates, x='HasMortgage', y='Default', color='blue')
plt.title('Default Rates by Mortgage Status')
plt.xlabel('Has Mortgage')
plt.ylabel('Default Rate')
plt.xticks(ticks=[0, 1], labels=['No Mortgage', 'Has Mortgage'])
plt.tight_layout()

st.subheader("Default Rates by Mortgage Status")
st.pyplot(plt.gcf())
