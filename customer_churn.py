# import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load model and preprocessing pipeline
def load_churn_model():
	model_path = 'Saved_Models/best_churn_model.joblib'
	preprocessor_path = 'Saved_Models/churn_preprocessor.joblib'
	if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
		st.error('Model or preprocessor not found. Please train and save them first.')
		st.stop()
	model = joblib.load(model_path)
	preprocessor = joblib.load(preprocessor_path)
	return model, preprocessor

def main():
	st.title(':blue-background[CUSTOMER CHURN PREDICTION]')
	st.subheader('Use this app to predict whether a Bank customer will churn or not, based on their profile, engagement metrics and spending behavior.')
	st.write(':blue-background[Please fill all the required fields from the list or by increasing/decreasing values using +/- signs.Then click **Predict churn** button below to see the prediction results.]')

	model, preprocessor = load_churn_model()

	# All required input fields for the model
	gender = st.selectbox('Customer Gender', ['Male', 'Female'])
	age = st.number_input('Customer Age', min_value=18, max_value=100, value=35)
	education = st.selectbox('Education Level', ['High School', 'Graduate', 'Uneducated', 'College', 'Post-Graduate', 'Doctorate', 'Unknown'])
	income = st.selectbox('Customer annual Income Category', ['Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +', 'Unknown'])
	marital_status = st.selectbox('Marital Status', ['Married', 'Single', 'Divorced', 'Unknown'])
	card_category = st.selectbox('Type of credit Card', ['Blue', 'Silver', 'Gold', 'Platinum'])
	dependent_count = st.number_input('Number of of Dependents', min_value=0, max_value=10, value=0)
	months_on_book = st.number_input('Number of Months the Account has been open', min_value=6, max_value=100, value=36)
	months_inactive_12_mon = st.number_input('Number of months the customer was inactive(Last 12 Months)', min_value=0, max_value=12, value=1)
	contacts_count_12_mon = st.number_input('Number of contacts (e.g., calls, emails) made with the customer in the last 12 months.)', min_value=0, max_value=20, value=2)
	total_relationship_count = st.number_input('Total number of products/services the customer uses with the bank.', min_value=1, max_value=10, value=3)
	credit_limit = st.number_input('Customer Credit Limit', min_value=1000.0, max_value=100000.0, value=10000.0, step=0.10)
	avg_open_to_buy = st.number_input('Average open-to-buy amount on the credit card', min_value=0.0, max_value=100000.0, value=5000.0, step=0.10)
	total_amt_chng_q4_q1 = st.number_input('Ratio of transaction amount change from Q1 to Q4', min_value=0.0, max_value=10.0, value=1.0, step=0.01)
	total_trans_amt = st.number_input('Total transaction amount over the last 12 months', min_value=0.0, max_value=100000.0, value=5000.0, step=0.100)
	total_trans_ct = st.number_input('Total transaction count over the last 12 months', min_value=0, max_value=500, value=40)
	total_ct_chng_q4_q1 = st.number_input('Ratio of transaction count change from Q1 to Q4', min_value=0.0, max_value=10.0, value=1.0, step=0.01)
	total_revolving_bal = st.number_input('Total revolving balance on the credit card', min_value=0, max_value=100000, value=1000)
	avg_utilization_ratio = st.number_input('Average credit card utilization ratio', min_value=0.0, max_value=1.0, value=0.2, step=0.01)
	# Naive Bayes Classifier features (if required, set to 0.0 by default)
	nb_1 = st.number_input('Predicted churn probability (class 1) from a pre-built Naive Bayes model.', min_value=0.0, max_value=1.0, value=0.0, step=0.01)
	nb_2 = st.number_input('Predicted churn probability (class 2) from a pre-built Naive Bayes model.', min_value=0.0, max_value=1.0, value=0.0, step=0.01)

	input_dict = {
		'Gender': gender,
		'Customer_Age': age,
		'Education_Level': education,
		'Income_Category': income,
		'Marital_Status': marital_status,
		'Card_Category': card_category,
		'Dependent_count': dependent_count,
		'Months_on_book': months_on_book,
		'Months_Inactive_12_mon': months_inactive_12_mon,
		'Contacts_Count_12_mon': contacts_count_12_mon,
		'Total_Relationship_Count': total_relationship_count,
		'Credit_Limit': credit_limit,
		'Avg_Open_To_Buy': avg_open_to_buy,
		'Total_Amt_Chng_Q4_Q1': total_amt_chng_q4_q1,
		'Total_Trans_Amt': total_trans_amt,
		'Total_Trans_Ct': total_trans_ct,
		'Total_Ct_Chng_Q4_Q1': total_ct_chng_q4_q1,
		'Total_Revolving_Bal': total_revolving_bal,
		'Avg_Utilization_Ratio': avg_utilization_ratio,
		'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1': nb_1,
		'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2': nb_2,
	}
	input_df = pd.DataFrame([input_dict])

	if st.button('Predict Churn'):
		try:
			X_processed = preprocessor.transform(input_df)
			pred = model.predict(X_processed)[0]
			pred_proba = model.predict_proba(X_processed)[0][1]
			st.write('-' * 25)
			st.write('**Prediction Results:**')
			st.success(f'{"  Customer is likely to churn" if pred == 1 else "Customer is not likely to churn"}')
			st.write('**Probability of a customer churning:**')
			st.write(f'Churn Probability: {pred_proba:.2%}')
		except Exception as e:
			st.error(f'Prediction error: {str(e)}')

if __name__ == '__main__':
	main()