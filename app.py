import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("student_performance_model.pkl")

with open("feature_schema.json", "r") as file:
    schema = json.load(file)

model = load_model()

st.title("🎓 Student Performance Predictor")
st.write(
    "Enter student information to estimate their exam score "
    "using a machine learning model."
)

st.info(
    "This tool provides an estimate based on historical data. "
    "It should support academic guidance, not determine a "
    "student's opportunities or potential."
)

with st.form("prediction_form"):
    st.subheader("Student Information")
    inputs = {}

    for column, details in schema.items():
        if details["type"] == "numeric":
            minimum = details["min"]
            maximum = details["max"]
            default = details["default"]

            inputs[column] = st.number_input(
                column.replace("_", " ").title(),
                min_value=float(minimum),
                max_value=float(maximum),
                value=float(default)
            )
        else:
            options = details["options"]

            if options:
                inputs[column] = st.selectbox(
                    column.replace("_", " ").title(),
                    options
                )
            else:
                inputs[column] = st.text_input(
                    column.replace("_", " ").title()
                )

    submitted = st.form_submit_button("Predict Exam Score")

if submitted:
    input_df = pd.DataFrame([inputs])

    try:
        prediction = float(model.predict(input_df)[0])

        st.subheader("Prediction Result")
        st.metric("Estimated Exam Score", f"{prediction:.2f}")

        st.caption(
            "This is a model-generated estimate, not a guaranteed result. "
            "Prediction reliability depends on the dataset and model quality."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
