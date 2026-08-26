import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Online Shopper Prediction",
    page_icon="🛒",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("online_shopper_model.pkl")

model = load_model()

st.title("🛒 Online Shoppers Purchasing Intention")
st.write("Predict whether an online shopper is likely to make a purchase.")

st.subheader("Enter Customer Browsing Details")

col1, col2 = st.columns(2)

with col1:
    Administrative = st.number_input("Administrative Pages Visited", min_value=0, value=0)
    Administrative_Duration = st.number_input("Administrative Duration", min_value=0.0, value=0.0)
    Informational = st.number_input("Informational Pages Visited", min_value=0, value=0)
    Informational_Duration = st.number_input("Informational Duration", min_value=0.0, value=0.0)
    ProductRelated = st.number_input("Product Related Pages", min_value=0, value=1)
    ProductRelated_Duration = st.number_input("Product Related Duration", min_value=0.0, value=10.0)
    BounceRates = st.number_input("Bounce Rate", min_value=0.0, max_value=1.0, value=0.0)
    ExitRates = st.number_input("Exit Rate", min_value=0.0, max_value=1.0, value=0.0)
    PageValues = st.number_input("Page Value", min_value=0.0, value=0.0)

with col2:
    SpecialDay = st.number_input("Special Day", min_value=0.0, max_value=1.0, value=0.0)

    Month = st.selectbox(
        "Month",
        ["Jan", "Feb", "Mar", "Apr", "May", "June",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )

    OperatingSystems = st.number_input("Operating System", min_value=1, value=1)
    Browser = st.number_input("Browser", min_value=1, value=1)
    Region = st.number_input("Region", min_value=1, value=1)
    TrafficType = st.number_input("Traffic Type", min_value=1, value=1)

    VisitorType = st.selectbox(
        "Visitor Type",
        ["Returning_Visitor", "New_Visitor", "Other"]
    )

    Weekend = st.selectbox("Weekend", [False, True])

if st.button("Predict Purchase Intention", use_container_width=True):

    input_data = pd.DataFrame({
        "Administrative": [Administrative],
        "Administrative_Duration": [Administrative_Duration],
        "Informational": [Informational],
        "Informational_Duration": [Informational_Duration],
        "ProductRelated": [ProductRelated],
        "ProductRelated_Duration": [ProductRelated_Duration],
        "BounceRates": [BounceRates],
        "ExitRates": [ExitRates],
        "PageValues": [PageValues],
        "SpecialDay": [SpecialDay],
        "Month": [Month],
        "OperatingSystems": [OperatingSystems],
        "Browser": [Browser],
        "Region": [Region],
        "TrafficType": [TrafficType],
        "VisitorType": [VisitorType],
        "Weekend": [Weekend]
    })

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("🎉 Prediction: Customer is likely to PURCHASE!")
    else:
        st.warning("❌ Prediction: Customer is NOT likely to purchase.")
