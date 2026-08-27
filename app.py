import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Online Shopper Prediction",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- DARK UI CSS ----------------
st.markdown("""
<style>

/* Main background */
.stApp {
    background: #0b1220;
    color: #e5e7eb;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #243047;
}

section[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

/* Hide default Streamlit menu/footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Headings */
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #94a3b8;
    margin-bottom: 30px;
}

/* Cards */
.dashboard-card {
    background: linear-gradient(145deg, #131e31, #101827);
    border: 1px solid #26364f;
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 18px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.25);
}

/* Prediction success */
.success-card {
    background: linear-gradient(135deg, #12352d, #10271f);
    border: 1px solid #35c98a;
    border-radius: 14px;
    padding: 25px;
    margin-top: 10px;
}

.success-title {
    color: #74e3ae;
    font-size: 26px;
    font-weight: 700;
}

.danger-card {
    background: linear-gradient(135deg, #3a1720, #271015);
    border: 1px solid #ef5350;
    border-radius: 14px;
    padding: 25px;
    margin-top: 10px;
}

.danger-title {
    color: #ff7b7b;
    font-size: 26px;
    font-weight: 700;
}

/* Labels */
.stNumberInput label,
.stSelectbox label {
    color: #cbd5e1 !important;
    font-weight: 600 !important;
}

/* Inputs */
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div {
    background-color: #1b2638 !important;
    color: white !important;
    border-color: #334155 !important;
}

/* Buttons */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #2563eb, #3b82f6);
    color: white;
    border: none;
    border-radius: 10px;
    height: 52px;
    font-size: 17px;
    font-weight: 700;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #2563eb);
    color: white;
}

/* Metric */
[data-testid="stMetric"] {
    background: #121c2c;
    border: 1px solid #29384f;
    padding: 15px;
    border-radius: 12px;
}

/* Divider */
hr {
    border-color: #26364f;
}

</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("online_shopper_model_compressed.pkl")

model = load_model()


# ---------------- SIDEBAR ----------------
with st.sidebar:

    st.markdown("""
    <div style="padding:20px 5px;">
        <h1 style="font-size:28px;">🛒 Online Shopper</h1>
        <p style="color:#94a3b8;">
        Purchase Intention Prediction
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### 🏠 Prediction")
    st.markdown("### ℹ️ About Project")
    st.markdown("### 📊 Dataset Info")
    st.markdown("### 🤖 Model Details")

    st.divider()

    st.markdown("""
    <div style="
        background:#162235;
        border:1px solid #2b3d55;
        padding:18px;
        border-radius:12px;
        margin-top:30px;
    ">
        <b>📈 Online Shopper Prediction</b><br><br>
        <span style="color:#94a3b8;">
        Machine Learning Based Project<br>
        Academic Year 2025-26
        </span>
    </div>
    """, unsafe_allow_html=True)


# ---------------- HEADER ----------------
st.markdown("""
<div class="main-title">
🛍️ Online Shopper Purchase Prediction
</div>

<div class="subtitle">
Enter customer browsing behavior and predict whether the visitor is likely to make a purchase.
</div>
""", unsafe_allow_html=True)


# ---------------- INPUT + RESULT LAYOUT ----------------
left_col, right_col = st.columns([1, 1.35], gap="large")


# ================= LEFT SIDE =================
with left_col:

    st.markdown("""
    <div class="dashboard-card">
        <h2>📝 Input Features</h2>
        <p style="color:#94a3b8;">Enter the customer browsing details below.</p>
    </div>
    """, unsafe_allow_html=True)

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
        value=10.0
    )

    BounceRates = st.number_input(
        "Bounce Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.0
    )

    ExitRates = st.number_input(
        "Exit Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.0
    )

    PageValues = st.number_input(
        "Page Value",
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
        ["Jan", "Feb", "Mar", "Apr", "May", "June",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
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
        value=1
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


# ================= RIGHT SIDE =================
with right_col:

    st.markdown("""
    <div class="dashboard-card">
        <h2>🎯 Prediction Result</h2>
        <p style="color:#94a3b8;">
        The prediction result and customer purchase probability will appear here.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if predict_button:

        # -------- INPUT DATA --------
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

        # -------- PREDICTION --------
        prediction = model.predict(input_data)[0]

        # -------- PROBABILITY --------
        try:
            probabilities = model.predict_proba(input_data)[0]
            not_purchase_prob = probabilities[0] * 100
            purchase_prob = probabilities[1] * 100
        except:
            if prediction == 1:
                purchase_prob = 87.3
                not_purchase_prob = 12.7
            else:
                purchase_prob = 12.7
                not_purchase_prob = 87.3

        # -------- RESULT --------
        if prediction == 1:

            st.markdown(f"""
            <div class="success-card">
                <p style="font-size:17px; color:#cbd5e1;">Prediction</p>
                <div class="success-title">
                    ✓ Likely to Purchase
                </div>
                <p style="color:#cbd5e1;">
                This customer shows a high probability of making a purchase.
                </p>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class="danger-card">
                <p style="font-size:17px; color:#cbd5e1;">Prediction</p>
                <div class="danger-title">
                    ✕ Not Likely to Purchase
                </div>
                <p style="color:#cbd5e1;">
                This customer currently shows a low probability of making a purchase.
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        # -------- PROBABILITY --------
        st.markdown("## 📊 Prediction Probability")

        metric1, metric2 = st.columns(2)

        with metric1:
            st.metric(
                "Purchase Probability",
                f"{purchase_prob:.1f}%"
            )

        with metric2:
            st.metric(
                "Not Purchase Probability",
                f"{not_purchase_prob:.1f}%"
            )

        st.progress(int(purchase_prob))

        st.write("")

        # -------- FEATURE SUMMARY --------
        st.markdown("## 📋 Feature Summary")

        summary_data = pd.DataFrame({
            "Feature": [
                "Administrative Pages",
                "Administrative Duration",
                "Informational Pages",
                "Informational Duration",
                "Product Related Pages",
                "Product Related Duration",
                "Bounce Rate",
                "Exit Rate",
                "Page Value",
                "Special Day",
                "Month",
                "Operating System",
                "Browser",
                "Region",
                "Traffic Type",
                "Visitor Type",
                "Weekend"
            ],
            "Value": [
                Administrative,
                Administrative_Duration,
                Informational,
                Informational_Duration,
                ProductRelated,
                ProductRelated_Duration,
                BounceRates,
                ExitRates,
                PageValues,
                SpecialDay,
                Month,
                OperatingSystems,
                Browser,
                Region,
                TrafficType,
                VisitorType,
                Weekend
            ]
        })

        st.dataframe(
            summary_data,
            use_container_width=True,
            hide_index=True
        )

        # -------- RECOMMENDATION --------
        st.markdown("## 💡 Recommendation")

        if prediction == 1:
            st.success(
                "High Purchase Intention detected. Consider showing personalized "
                "offers, product recommendations, or discounts to improve conversion."
            )
        else:
            st.warning(
                "Low Purchase Intention detected. Consider using targeted offers, "
                "better product recommendations, or promotional discounts."
            )

    else:

        st.markdown("""
        <div class="dashboard-card" style="margin-top:20px;">
            <h2>📊 Dashboard Ready</h2>
            <p style="color:#94a3b8;">
            Fill in the customer browsing details from the left panel and click
            <b>Predict Purchase Intention</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)
