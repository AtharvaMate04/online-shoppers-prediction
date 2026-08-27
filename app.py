import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Online Shopper Prediction",
    page_icon="🛒",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: #0e1726;
        color: white;
    }

    section[data-testid="stSidebar"] {
        background: #162238;
    }

    .main-title {
        font-size: 42px;
        font-weight: 700;
        color: #f1f5f9;
    }

    .subtitle {
        font-size: 18px;
        color: #b8c1d1;
        margin-bottom: 25px;
    }

    .card {
        background: #162238;
        padding: 22px;
        border-radius: 12px;
        border: 1px solid #30425f;
        margin-bottom: 20px;
    }

    .result-success {
        background: #102d27;
        border: 1px solid #38d996;
        padding: 25px;
        border-radius: 12px;
    }

    .result-fail {
        background: #351d25;
        border: 1px solid #ff5c6c;
        padding: 25px;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("online_shopper_model_compressed.pkl")

model = load_model()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:

    st.markdown("# 🛒 Online Shopper")
    st.caption("Predict purchasing intention using Machine Learning")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Prediction",
            "ℹ️ About Project",
            "🗄️ Dataset Info",
            "🤖 Model Details"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("""
    ### 📊 Project Info

    Built with ❤️ using Streamlit

    Academic Project  
    **2025–26**
    """)

# ==================================================
# PREDICTION PAGE
# ==================================================
if page == "🏠 Prediction":

    st.markdown(
        '<div class="main-title">🛍️ Online Shopper Purchase Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Enter the customer behavior details and predict whether the user will make a purchase.</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1, 1.3])

    # ---------------- INPUT SECTION ----------------
    with col1:

        st.markdown("## 📝 Input Features")
        st.write("Enter the customer browsing details below:")

        Administrative = st.number_input(
            "Administrative Pages Visited",
            min_value=0,
            value=0
        )

        Administrative_Duration = st.number_input(
            "Administrative Duration (sec)",
            min_value=0.0,
            value=0.0
        )

        Informational = st.number_input(
            "Informational Pages Visited",
            min_value=0,
            value=0
        )

        Informational_Duration = st.number_input(
            "Informational Duration (sec)",
            min_value=0.0,
            value=0.0
        )

        ProductRelated = st.number_input(
            "Product Related Pages",
            min_value=0,
            value=1
        )

        ProductRelated_Duration = st.number_input(
            "Product Related Duration (sec)",
            min_value=0.0,
            value=75.0
        )

        BounceRates = st.number_input(
            "Bounce Rates",
            min_value=0.0,
            max_value=1.0,
            value=0.02
        )

        ExitRates = st.number_input(
            "Exit Rates",
            min_value=0.0,
            max_value=1.0,
            value=0.02
        )

        PageValues = st.number_input(
            "Page Values",
            min_value=0.0,
            value=0.0
        )

        SpecialDay = st.number_input(
            "Special Day",
            min_value=0.0,
            max_value=1.0,
            value=0.0
        )

        Month = st.selectbox(
            "Month",
            ["Jan", "Feb", "Mar", "Apr", "May",
             "June", "Jul", "Aug", "Sep", "Oct",
             "Nov", "Dec"]
        )

        OperatingSystems = st.number_input(
            "Operating System",
            min_value=1,
            value=1
        )

        Browser = st.number_input(
            "Browser",
            min_value=1,
            value=1
        )

        Region = st.number_input(
            "Region",
            min_value=1,
            value=1
        )

        TrafficType = st.number_input(
            "Traffic Type",
            min_value=1,
            value=2
        )

        VisitorType = st.selectbox(
            "Visitor Type",
            ["Returning_Visitor", "New_Visitor", "Other"]
        )

        Weekend = st.selectbox(
            "Weekend",
            [False, True]
        )

        predict_button = st.button(
            "▶ Predict Purchase Intention",
            use_container_width=True
        )

    # ---------------- RESULT SECTION ----------------
    with col2:

        st.markdown("## 🎯 Prediction Result")

        if predict_button:

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

            # Probability
            try:
                probability = model.predict_proba(input_data)[0][1] * 100
            except:
                probability = 50.0

            if prediction == 1:

                st.success("### ✅ Likely to Purchase")
                st.write(
                    "This user is likely to make a purchase based on the provided browsing information."
                )

                st.markdown("## 📊 Prediction Probability")

                c1, c2 = st.columns(2)

                with c1:
                    st.metric("Purchase Probability", f"{probability:.1f}%")

                with c2:
                    st.metric(
                        "Not Purchase Probability",
                        f"{100-probability:.1f}%"
                    )

                st.progress(int(probability))

                st.success(
                    "✅ High Purchase Intention: This user shows strong buying interest."
                )

                st.markdown("## 💡 Recommendation")

                st.info(
                    "Consider showing personalized offers or product recommendations to improve conversion."
                )

            else:

                st.warning("### ❌ Not Likely to Purchase")
                st.write(
                    "This user currently shows a lower probability of making a purchase."
                )

                st.markdown("## 📊 Prediction Probability")

                c1, c2 = st.columns(2)

                with c1:
                    st.metric(
                        "Purchase Probability",
                        f"{probability:.1f}%"
                    )

                with c2:
                    st.metric(
                        "Not Purchase Probability",
                        f"{100-probability:.1f}%"
                    )

                st.progress(int(probability))

                st.markdown("## 💡 Recommendation")

                st.warning(
                    "Try personalized discounts, product recommendations, or promotional offers."
                )

            # Feature Summary
            st.markdown("## 📄 Feature Summary")

            summary_data = pd.DataFrame({
                "Feature": input_data.columns,
                "Value": input_data.iloc[0].astype(str).values
            })

            st.dataframe(
                summary_data,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "👈 Enter the customer details and click Predict Purchase Intention."
            )


# ==================================================
# ABOUT PROJECT PAGE
# ==================================================
elif page == "ℹ️ About Project":

    st.title("ℹ️ About Project")

    st.markdown("""
    ## Online Shopper Purchasing Intention Prediction

    This project uses Machine Learning to predict whether an online shopper is likely to make a purchase.

    ### 🎯 Objective
    The main objective is to analyze customer browsing behavior and predict purchasing intention.

    ### 🛠️ Technologies Used
    - Python
    - Streamlit
    - Machine Learning
    - Pandas
    - Scikit-learn
    - Joblib

    ### 👨‍🎓 Student Details
    **Name:** Mate Atharva Vinayak  
    **Branch:** Information Technology  
    **Academic Year:** 2025–26  

    **College:** Matoshri Asarabai Institute of Technology and Research Centre, Eklahare, Nashik
    """)


# ==================================================
# DATASET PAGE
# ==================================================
elif page == "🗄️ Dataset Info":

    st.title("🗄️ Dataset Information")

    st.write("""
    The dataset contains information about the browsing behavior of online shoppers.
    """)

    dataset_info = pd.DataFrame({
        "Feature": [
            "Administrative",
            "Informational",
            "ProductRelated",
            "BounceRates",
            "ExitRates",
            "PageValues",
            "SpecialDay",
            "Month",
            "VisitorType",
            "Weekend"
        ],
        "Description": [
            "Number of administrative pages visited",
            "Number of informational pages visited",
            "Number of product related pages visited",
            "Bounce rate of visitor",
            "Exit rate of visitor",
            "Average page value",
            "Closeness to special day",
            "Month of visit",
            "Type of visitor",
            "Weekend or weekday visit"
        ]
    })

    st.dataframe(
        dataset_info,
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# MODEL DETAILS PAGE
# ==================================================
elif page == "🤖 Model Details":

    st.title("🤖 Model Details")

    st.markdown("""
    ### Machine Learning Model

    The trained Machine Learning model analyzes customer browsing behavior and predicts whether the customer is likely to complete a purchase.

    ### Input Features
    - Administrative Pages
    - Informational Pages
    - Product Related Pages
    - Bounce Rate
    - Exit Rate
    - Page Value
    - Month
    - Visitor Type
    - Weekend
    - Other browsing features

    ### Output

    **1 → Likely to Purchase**

    **0 → Not Likely to Purchase**
    """)
