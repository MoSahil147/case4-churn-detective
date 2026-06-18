import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import plotly.express as px
import shap
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Churn Detective", layout="wide")

# cache_resource keeps the model in memory across user sessions — avoids reloading on every page refresh
@st.cache_resource
def load_model():
    # rb = read binary, pickle converts the saved file back into a Python object
    with open("outputs/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("outputs/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    return model, feature_names

# cache_data caches the dataframe — Streamlit re-runs the whole script on every interaction,
# so without caching it would re-read the CSV every single time
@st.cache_data
def load_data():
    return pd.read_csv("data/case4_telecom_churn.csv")

@st.cache_data
def load_segments():
    # churner_segments.csv was saved by the notebook after KMeans clustering
    return pd.read_csv("outputs/churner_segments.csv")

@st.cache_resource
def build_encoders(_df):
    # Re-fit a LabelEncoder per categorical column using the same dataset as the notebook
    # This ensures user input on the Predict page gets encoded the exact same way as training data
    # The underscore prefix on _df tells Streamlit not to hash this argument (DataFrames aren't hashable)
    cat_cols = _df.select_dtypes(include="str").columns.tolist() # finding the columns which have str
    cat_cols = [c for c in cat_cols if c != "customer_id"]  # customer_id is not a feature
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(_df[col])        # learn all possible category values for this column, the underscore before tells not to hash the DF
        encoders[col] = le
    return encoders

# Load everything once when the app starts
model, feature_names = load_model()
df = load_data()
segments_df = load_segments()
encoders = build_encoders(df)

# Global constants used across pages
CHURN_RATE = df["churned"].mean()   # 36.2% — baseline everything is compared against
MODEL_AUC  = 0.7226                 # XGBoost ROC-AUC from the notebook test set

# Sidebar navigation — st.sidebar.radio renders a set of radio buttons in the left panel
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "EDA Explorer", "Predict", "Segments"],
    index=0, # default at Overview
)

# Each page is just an if/elif block — Streamlit re-runs the whole script every time
# the user clicks something, and only the active page's code runs

if page == "Overview":
    st.title("Churn Detective — Case 4")
    st.markdown("**A mid-sized telecom is bleeding postpaid customers. This dashboard helps the CMO know *who* is about to leave and *why*.**")
    st.divider()

    # st.columns splits the page into equal-width columns side by side
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{len(df):,}")
    c2.metric("Overall Churn Rate", f"{CHURN_RATE:.1%}")
    c3.metric("XGBoost ROC-AUC", f"{MODEL_AUC:.4f}")
    c4.metric("Industry Benchmark", "1.5% / month")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Churn Distribution")
        st.image("outputs/figures/churn_rate_pie.png")

    with col_b:
        st.subheader("Model Comparison — ROC Curves")
        st.image("outputs/figures/roc_curves.png")

    st.divider()
    st.subheader("Top Churn Drivers (SHAP — Global)")
    st.image("outputs/figures/shap_beeswarm.png", use_container_width=True)
    st.caption("Each dot = one customer. Red = high feature value. Dots right of centre push toward churn.")

    st.divider()
    st.subheader("Cost-Aware Threshold Tuning")
    st.image("outputs/figures/threshold_tuning.png", use_container_width=True)
    st.caption("Red line = optimal threshold that maximises expected revenue saved ($100 offer cost, $670 LTV saved).")


elif page == "EDA Explorer":
    st.title("EDA Explorer")
    st.markdown("Pick any feature to see how it relates to churn.")

    num_cols = ["tenure_months", "monthly_charges", "total_charges",
                "support_calls_3mo", "avg_data_gb_3mo", "late_payments_6mo", "plan_changes_6mo"]
    cat_cols = ["contract_type", "internet_service", "payment_method",
                "online_security", "tech_support", "paperless_billing"]

    feature_type = st.radio("Feature type", ["Numeric", "Categorical"], horizontal=True)

    if feature_type == "Numeric":
        col = st.selectbox("Select numeric feature", num_cols)
        # histnorm="probability density" normalises both groups so churners and retained
        # are directly comparable even though retained customers outnumber churners 2:1
        fig = px.histogram(
            df, 
            x=col, 
            color=df["churned"].map({0: "Retained", 1: "Churned"}),
            barmode="overlay", 
            histnorm="probability density",
            color_discrete_map={"Retained": "#4CAF50", "Churned": "#F44336"},
            title=f"{col} distribution by churn status",
            labels={"color": "Status"},
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        col = st.selectbox("Select categorical feature", cat_cols)
        # Calculate churn rate per category value, sort highest to lowest
        churn_by = df.groupby(col)["churned"].mean().reset_index()
        churn_by.columns = [col, "churn_rate"] # renaming churned to churn_rate
        churn_by = churn_by.sort_values("churn_rate", ascending=False)
        fig = px.bar(
            churn_by, x=col, y="churn_rate",
            title=f"Churn rate by {col}",
            color="churn_rate",
            color_continuous_scale="RdYlGn_r",   # red = high churn, green = low churn
            labels={"churn_rate": "Churn Rate"},
        )
        # Red dashed line shows the overall average so you can see which categories are above/below baseline
        fig.add_hline(y=CHURN_RATE, 
                      line_dash="dash", 
                      line_color="red",
                      annotation_text=f"Avg {CHURN_RATE:.1%}")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    # Static pre-saved figures from the notebook — quicker to render than recomputing
    st.subheader("Pre-saved EDA figures")
    img_choice = st.selectbox("View", [
        "Numeric distributions", "Categorical churn rates", "Correlation heatmap", "Confusion matrices"
    ])
    img_map = {
        "Numeric distributions": "outputs/figures/numeric_distributions.png",
        "Categorical churn rates": "outputs/figures/categorical_churn_rates.png",
        "Correlation heatmap": "outputs/figures/correlation_heatmap.png",
        "Confusion matrices": "outputs/figures/confusion_matrices.png",
    }
    st.image(img_map[img_choice], use_container_width=True)


elif page == "Predict":
    st.title("Live Churn Predictor")
    st.markdown("Adjust the customer profile below to get an instant churn probability and a SHAP explanation.")

    # st.form groups all inputs together — the model only runs when the user clicks Submit,
    # not on every individual slider move (which would be very slow with SHAP)
    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Contract & Billing**")
            contract_type     = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
            payment_method    = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (auto)", "Credit card (auto)"])
            paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])
            monthly_charges   = st.slider("Monthly charges ($)", 20.0, 120.0, 70.0, 1.0) # default at 70, 1 step move at a time
            tenure_months     = st.slider("Tenure (months)", 1, 72, 24)

        with c2:
            st.markdown("**Services**")
            internet_service = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
            online_security  = st.selectbox("Online security", ["No", "Yes", "No internet service"])
            tech_support     = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
            streaming_tv     = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            phone_service    = st.selectbox("Phone service", ["Yes", "No"])
            multiple_lines   = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])

        with c3:
            st.markdown("**Behaviour & Demographics**")
            support_calls_3mo = st.slider("Support calls (3 mo)", 0, 10, 1)
            late_payments_6mo = st.slider("Late payments (6 mo)", 0, 6, 0)
            plan_changes_6mo  = st.slider("Plan changes (6 mo)", 0, 5, 0)
            avg_data_gb_3mo   = st.slider("Avg data GB (3 mo)", 0.0, 100.0, 30.0, 1.0)
            senior_citizen    = st.selectbox("Senior citizen", [0, 1])
            partner           = st.selectbox("Partner", ["No", "Yes"])
            dependents        = st.selectbox("Dependents", ["No", "Yes"])

        submitted = st.form_submit_button("Predict Churn Probability", use_container_width=True)

    if submitted:
        # total_charges is derived from tenure × monthly — same calculation as the original dataset
        total_charges = monthly_charges * tenure_months

        # Build a one-row dictionary with all 19 feature values
        raw = {
            "tenure_months":     tenure_months,
            "contract_type":     contract_type,
            "monthly_charges":   monthly_charges,
            "total_charges":     total_charges,
            "internet_service":  internet_service,
            "online_security":   online_security,
            "tech_support":      tech_support,
            "streaming_tv":      streaming_tv,
            "payment_method":    payment_method,
            "paperless_billing": paperless_billing,
            "senior_citizen":    senior_citizen,
            "partner":           partner,
            "dependents":        dependents,
            "phone_service":     phone_service,
            "multiple_lines":    multiple_lines,
            "support_calls_3mo": support_calls_3mo,
            "avg_data_gb_3mo":   avg_data_gb_3mo,
            "late_payments_6mo": late_payments_6mo,
            "plan_changes_6mo":  plan_changes_6mo,
        }

        input_df = pd.DataFrame([raw]) # building one row DF from users input similar to how in training data

        # Encode categorical text values to integers using the same encoders as training
        # e.g. "Month-to-month" → 0, "One year" → 1, "Two year" → 2
        for col, le in encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col])

        # Reorder columns to exactly match the order the model was trained on
        input_df = input_df[feature_names]

        # predict_proba returns [[prob_retain, prob_churn]] — we take index 1 for churn probability
        prob = model.predict_proba(input_df)[0, 1]

        st.divider()
        # Colour-coded risk label: red for high, orange for medium, green for low
        risk_label = "HIGH RISK" if prob >= 0.5 else ("MEDIUM RISK" if prob >= 0.3 else "LOW RISK")
        color = "#F44336" if prob >= 0.5 else ("#FF9800" if prob >= 0.3 else "#4CAF50")

        st.markdown(f"### Churn Probability: <span style='color:{color}; font-size:2rem'>{prob:.1%} — {risk_label}</span>", unsafe_allow_html=True)
        st.progress(float(prob))

        st.divider()
        st.subheader("Why? — SHAP Explanation")

        # TreeExplainer computes exact SHAP values for tree models — faster than KernelExplainer
        # shap_vals[0] is the explanation for the first (and only) row in our single-customer input
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer(input_df)

        fig, ax = plt.subplots(figsize=(9, 4))
        shap.plots.waterfall(shap_vals[0], show=False)   # show=False lets us control when to display
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()   # release memory so matplotlib doesn't accumulate open figures

        st.caption("Red bars push toward churn. Blue bars pull away from churn. Bar length = how much that feature mattered for this customer.")


elif page == "Segments":
    st.title("Churner Segments")
    st.markdown("5,505 predicted churners grouped into 3 actionable segments. Each needs a different retention play.")

    # Build a summary table: one row per segment with average feature values and size
    summary = segments_df.groupby("segment_label").agg( # agg calculates the 5 things at once per segment
        size=("segment_label", "count"),
        avg_charges=("monthly_charges", "mean"),
        avg_support=("support_calls_3mo", "mean"),
        avg_tenure=("tenure_months", "mean"),
        avg_late=("late_payments_6mo", "mean"),
    ).round(2).reset_index()
    summary["pct"] = (summary["size"] / len(segments_df) * 100).round(1)

    # One metric card per segment, side by side
    cols = st.columns(len(summary))
    for i, row in summary.iterrows():
        cols[i].metric(row["segment_label"], f"{row['size']:,} customers ({row['pct']}%)")
        cols[i].caption(f"Avg charges: ${row['avg_charges']:.0f} | Support calls: {row['avg_support']:.1f} | Late pmts: {row['avg_late']:.2f}")

    st.divider()

    # Interactive scatter: x = charges, y = support calls, bubble size = tenure
    # This lets the CMO visually explore how the 3 segments differ
    st.subheader("Monthly Charges vs Support Calls (bubble size = tenure)")
    fig = px.scatter(
        segments_df,
        x="monthly_charges",
        y="support_calls_3mo",
        color="segment_label",
        size="tenure_months",
        hover_data=["late_payments_6mo", "contract_type"],   # extra info on hover
        color_discrete_sequence=px.colors.qualitative.Set2,
        title="Churner Segments",
        labels={"monthly_charges": "Monthly Charges ($)", "support_calls_3mo": "Support Calls (3 mo)"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Retention playbook — one offer per segment based on the root cause identified in the analysis
    st.subheader("Retention Playbook")
    offers = pd.DataFrame([
        {
            "Segment": "Contract-Free At-Risk",
            "Root Cause": "Month-to-month — leaving is frictionless",
            "Offer": "12-month plan + 15% discount + free add-on",
            "Expected Impact": "~$424K recoverable at 20% conversion",
        },
        {
            "Segment": "Payment-Delinquent",
            "Root Cause": "1.25 avg late payments — billing friction",
            "Offer": "Flexible payment date + late fee waiver + auto-pay $10/mo off",
            "Expected Impact": "Reduce billing dropout before cancellation",
        },
        {
            "Segment": "Service-Frustrated",
            "Root Cause": "7.19 avg support calls — unresolved service issues",
            "Offer": "Dedicated rep + 48h SLA guarantee + 1 month free if unresolved",
            "Expected Impact": "Fix the service, keep the customer",
        },
    ])
    st.dataframe(offers, use_container_width=True, hide_index=True)

    st.divider()

    # Side-by-side SHAP waterfall examples from the notebook
    # Shows what a high-risk vs low-risk customer actually looks like feature-by-feature
    st.subheader("SHAP Waterfall Examples")
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**High-risk churner**")
        st.image("outputs/figures/shap_waterfall_churner.png", use_container_width=True)
    with col_r:
        st.markdown("**Low-risk retained customer**")
        st.image("outputs/figures/shap_waterfall_retained.png", use_container_width=True)
