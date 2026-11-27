"""Small Streamlit interface for the German Credit Risk API."""

import os

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

CHOICES = {
    "status_checking": ["A11", "A12", "A13", "A14"],
    "credit_history": ["A30", "A31", "A32", "A33", "A34"],
    "purpose": ["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A48", "A49", "A410"],
    "savings": ["A61", "A62", "A63", "A64", "A65"],
    "employment_since": ["A71", "A72", "A73", "A74", "A75"],
    "personal_status_sex": ["A91", "A92", "A93", "A94", "A95"],
    "other_debtors": ["A101", "A102", "A103"],
    "property": ["A121", "A122", "A123", "A124"],
    "other_installment_plans": ["A141", "A142", "A143"],
    "housing": ["A151", "A152", "A153"],
    "job": ["A171", "A172", "A173", "A174"],
    "telephone": ["A191", "A192"],
    "foreign_worker": ["A201", "A202"],
}

CATEGORY_LABELS = {
    "status_checking": {
        "A11": "Below 0 DM", "A12": "0 to under 200 DM",
        "A13": "200 DM or more / salary assignment", "A14": "No checking account",
    },
    "credit_history": {
        "A30": "No previous credit / repaid on time",
        "A31": "All credits at this bank repaid on time",
        "A32": "Existing credits repaid on time so far",
        "A33": "Late payments in the past",
        "A34": "Critical account / other credits elsewhere",
    },
    "purpose": {
        "A40": "New car", "A41": "Used car", "A42": "Furniture or equipment",
        "A43": "Radio or television", "A44": "Domestic appliances", "A45": "Repairs",
        "A46": "Education", "A48": "Retraining", "A49": "Business", "A410": "Other",
    },
    "savings": {
        "A61": "Below 100 DM", "A62": "100 to under 500 DM",
        "A63": "500 to under 1,000 DM", "A64": "1,000 DM or more",
        "A65": "No savings account / unknown",
    },
    "employment_since": {
        "A71": "Unemployed", "A72": "Employed under 1 year",
        "A73": "Employed 1 to under 4 years", "A74": "Employed 4 to under 7 years",
        "A75": "Employed 7 years or more",
    },
    "personal_status_sex": {
        "A91": "Male, divorced or separated", "A92": "Female, divorced/separated/married",
        "A93": "Male, single", "A94": "Male, married or widowed", "A95": "Female, single",
    },
    "other_debtors": {"A101": "None", "A102": "Co-applicant", "A103": "Guarantor"},
    "property": {
        "A121": "Real estate", "A122": "Building savings or life insurance",
        "A123": "Car or other property", "A124": "No property / unknown",
    },
    "other_installment_plans": {"A141": "Bank", "A142": "Retail store", "A143": "None"},
    "housing": {"A151": "Rent", "A152": "Own", "A153": "Free housing"},
    "job": {
        "A171": "Unemployed or unskilled non-resident", "A172": "Unskilled resident",
        "A173": "Skilled employee or official",
        "A174": "Management, self-employed, highly qualified, or officer",
    },
    "telephone": {"A191": "No telephone", "A192": "Telephone registered in applicant's name"},
    "foreign_worker": {"A201": "Yes", "A202": "No"},
}


def category_label(field: str, code: str) -> str:
    """Show a readable category while keeping the dataset code internal."""
    return CATEGORY_LABELS[field][code]


st.set_page_config(page_title="German Credit Risk Demo", page_icon="💳")
st.title("German Credit Risk Demo")
st.caption("Enter an application to get a prediction from the Practical Task 6 API.")
st.info("This is a classroom demonstration on historical data, not financial advice.")

with st.form("credit_application"):
    st.subheader("Account and credit history")
    col1, col2 = st.columns(2)
    with col1:
        status_checking = st.selectbox("Checking account status", CHOICES["status_checking"], format_func=lambda code: category_label("status_checking", code))
        credit_history = st.selectbox("Credit history", CHOICES["credit_history"], format_func=lambda code: category_label("credit_history", code))
        purpose = st.selectbox("Credit purpose", CHOICES["purpose"], index=3, format_func=lambda code: category_label("purpose", code))
        savings = st.selectbox("Savings", CHOICES["savings"], index=4, format_func=lambda code: category_label("savings", code))
        employment_since = st.selectbox("Employment duration", CHOICES["employment_since"], index=4, format_func=lambda code: category_label("employment_since", code))
    with col2:
        duration_months = st.number_input("Duration (months)", min_value=1, value=6)
        credit_amount = st.number_input("Credit amount", min_value=1, value=1169)
        installment_rate = st.selectbox("Installment rate", [1, 2, 3, 4], index=3)
        personal_status_sex = st.selectbox("Personal status", CHOICES["personal_status_sex"], index=2, format_func=lambda code: category_label("personal_status_sex", code))
        other_debtors = st.selectbox("Other debtors", CHOICES["other_debtors"], format_func=lambda code: category_label("other_debtors", code))

    st.subheader("Personal and financial details")
    col3, col4 = st.columns(2)
    with col3:
        present_residence = st.selectbox("Years at current residence", [1, 2, 3, 4], index=3)
        property_code = st.selectbox("Property", CHOICES["property"], format_func=lambda code: category_label("property", code))
        age = st.number_input("Age", min_value=18, value=67)
        other_installment_plans = st.selectbox("Other installment plans", CHOICES["other_installment_plans"], format_func=lambda code: category_label("other_installment_plans", code))
        housing = st.selectbox("Housing", CHOICES["housing"], index=1, format_func=lambda code: category_label("housing", code))
    with col4:
        number_credits = st.number_input("Existing credits at this bank", min_value=1, value=2)
        job = st.selectbox("Job", CHOICES["job"], index=2, format_func=lambda code: category_label("job", code))
        people_liable = st.number_input("People liable for maintenance", min_value=1, value=1)
        telephone = st.selectbox("Telephone", CHOICES["telephone"], index=1, format_func=lambda code: category_label("telephone", code))
        foreign_worker = st.selectbox("Foreign worker", CHOICES["foreign_worker"], format_func=lambda code: category_label("foreign_worker", code))

    submitted = st.form_submit_button("Predict risk")

if submitted:
    payload = {
        "status_checking": status_checking,
        "duration_months": int(duration_months),
        "credit_history": credit_history,
        "purpose": purpose,
        "credit_amount": int(credit_amount),
        "savings": savings,
        "employment_since": employment_since,
        "installment_rate": installment_rate,
        "personal_status_sex": personal_status_sex,
        "other_debtors": other_debtors,
        "present_residence": present_residence,
        "property": property_code,
        "age": int(age),
        "other_installment_plans": other_installment_plans,
        "housing": housing,
        "number_credits": int(number_credits),
        "job": job,
        "people_liable": int(people_liable),
        "telephone": telephone,
        "foreign_worker": foreign_worker,
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        if result["prediction"] == 0:
            st.success(f"Prediction: {result['risk_label']}")
        else:
            st.warning(f"Prediction: {result['risk_label']}")
    except requests.RequestException as error:
        st.error(f"Could not reach the prediction API at {API_URL}. Start the API and try again. ({error})")
