#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
STEP 3: The Streamlit Web App (the "prototype" deliverable)

This loads the trained pipeline saved in Step 2, lets someone type in a
company's financial figures, and shows a bankruptcy risk result.

This version includes three additions beyond a plain text result:
  1. A bar chart comparing the three experts' opinions and the final result
  2. A feature importance chart, showing which figures mattered most overall
  3. A plain-language "About the models" expandable section

To run this for real:  streamlit run step3_streamlit_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

with open("trained_pipeline_final.joblib", "rb") as f:
  pipeline_bundle = pickle.load(f)
scaler = pipeline_bundle["scaler"]
all_feature_columns = pipeline_bundle["all_feature_columns"]
selected_features = pipeline_bundle["selected_features"]
feature_importances = pipeline_bundle["feature_importances"]
rf_model = pipeline_bundle["rf_model"]
gb_model = pipeline_bundle["gb_model"]
knn_model = pipeline_bundle["knn_model"]
meta_model = pipeline_bundle["meta_model"]

FEATURE_LABELS = {
    "X1": "Current Assets", "X2": "Cost of Goods Sold",
    "X3": "Depreciation and Amortization", "X4": "EBITDA",
    "X5": "Inventory", "X6": "Net Income", "X7": "Total Receivables",
    "X8": "Market Value", "X9": "Net Sales", "X10": "Total Assets",
    "X11": "Total Long-Term Debt", "X12": "EBIT", "X13": "Gross Profit",
    "X14": "Total Current Liabilities", "X15": "Retained Earnings",
    "X16": "Total Revenue", "X17": "Total Liabilities",
    "X18": "Total Operating Expenses",
}

# The 18 figures grouped the way they appear on a real financial statement,
# rather than in raw dataset order. This makes the form far easier for user evaluators to understand
FEATURE_GROUPS = {
    "Assets \u2013 what the company owns": ["X1", "X5", "X7", "X10"],
    "Liabilities \u2013 what the company owes": ["X11", "X14", "X17"],
    "Income and profitability": ["X6", "X9", "X13", "X16", "X4", "X12"],
    "Costs and other measures": ["X2", "X18", "X3", "X8", "X15"],
}

THRESHOLD_OPTIONS = {
    "Standard - fewer false alarms (threshold 0.40)": 0.40,
    "Sensitive - catches more real bankruptcies (threshold 0.30)": 0.30,
}

st.set_page_config(page_title="Bankruptcy Risk Prototype", layout="centered")
st.title("Corporate Bankruptcy Risk Prototype")
st.write(
    "This is an MSc research prototype. Enter a company's financial "
    "figures below and the stacking ensemble will estimate its "
    "bankruptcy risk. This is for demonstration purposes only and "
    "is not financial advice."
)


# ADDITION 3: a plain-language explanation of the models used,
# hidden by default so it does not clutter the page, but available
# to anyone who wants to understand what is actually happening.

with st.expander("About the models used in this prediction"):
    st.markdown("""
    This tool does not rely on a single model. Instead, it asks **three
    different "expert" models** for their opinion, then a fourth
    **"manager" model** decides how much to trust each one before giving
    a final answer.

    - **Random Forest** builds hundreds of simple decision rules and lets
      them vote on the outcome.
    - **Gradient Boosting** builds a chain of decision rules, where each
      new rule tries to fix the mistakes of the ones before it.
    - **k-Nearest Neighbours** looks at the companies in the training data
      that are most financially similar to the one entered, and copies
      whatever happened to most of them.
    - **The meta-learner (manager)** is a simpler model that learns, from
      past results, how much to trust each of the three experts above,
      and combines their opinions into one final risk estimate.

    Combining several different-thinking models like this, rather than
    relying on just one, is called a **stacking ensemble**, and it is the
    central idea this research project tests.
    """)

st.subheader("Choose a sensitivity setting")
chosen_label = st.radio(
    "Two settings were tested and validated for this project:",
    list(THRESHOLD_OPTIONS.keys()),
)
chosen_threshold = THRESHOLD_OPTIONS[chosen_label]
st.caption(
    "The 'Standard' setting raises fewer false alarms but misses more real "
    "bankruptcies. The 'Sensitive' setting catches more real bankruptcies "
    "but raises more false alarms. Neither setting is perfect - this is a "
    "genuine trade-off, explained further in the accompanying report."
)


# ADDITION 2: feature importance chart - shown up front, since it
# describes the model in general rather than any one prediction.

st.subheader("Which financial figures matter most, overall")
st.caption(
    "This chart is based on the training data as a whole, not the specific "
    "company entered below - it shows what the model generally pays most "
    "attention to when judging bankruptcy risk."
)
importance_df = pd.DataFrame({
    "Feature": [f"{k} - {FEATURE_LABELS.get(k, '')}" for k in feature_importances.keys()],
    "Importance": list(feature_importances.values()),
}).sort_values("Importance", ascending=True)

fig_imp, ax_imp = plt.subplots(figsize=(7, 4.5))
ax_imp.barh(importance_df["Feature"], importance_df["Importance"], color="#3B4A9E")
ax_imp.set_xlabel("Importance score")
ax_imp.set_title("Top Financial Figures Used by the Model")
plt.tight_layout()
st.pyplot(fig_imp)

st.subheader("Enter the company's financial figures")
st.caption("Figures are in the same units as the training data (millions of US dollars).")


# ADDITION 4: example companies, drawn from the real, unseen test
# set, so a UAT participant does not need to type in 18 numbers by
# hand. Selecting an example fills every field automatically; the
# true outcome (bankrupt or not) is revealed only after a prediction
# is made, so participants can compare the model's answer to what
# actually happened.

EXAMPLE_COMPANIES = {
    "-- Enter your own figures --": None,
    "Example A": {
        "X1": 502.627, "X2": 566.984, "X3": 32.812, "X4": 15.625, "X5": 137.076,
        "X6": -316.121, "X7": 256.910, "X8": 103.1365, "X9": 711.359, "X10": 767.024,
        "X11": 302.916, "X12": -17.187, "X13": 144.375, "X14": 302.518, "X15": -127.622,
        "X16": 711.359, "X17": 671.739, "X18": 695.734,
    },
    "Example B": {
        "X1": 381.832, "X2": 712.950, "X3": 42.051, "X4": 158.358, "X5": 103.268,
        "X6": 79.174, "X7": 124.553, "X8": 2450.4519, "X9": 1084.224, "X10": 867.228,
        "X11": 0.904, "X12": 116.307, "X13": 371.274, "X14": 119.044, "X15": 664.940,
        "X16": 1084.224, "X17": 184.906, "X18": 925.866,
    },
    "Example C": {
        "X1": 49.583, "X2": 94.356, "X3": 8.459, "X4": 1.640, "X5": 18.269,
        "X6": -7.170, "X7": 28.127, "X8": 28.7345, "X9": 111.212, "X10": 102.237,
        "X11": 2.332, "X12": -6.819, "X13": 16.856, "X14": 47.817, "X15": 28.468,
        "X16": 111.212, "X17": 58.048, "X18": 109.572,
    },
}
# The true outcome for each example, revealed only after a prediction is made,
# never shown up front, so it cannot bias a participant's expectation.
EXAMPLE_TRUE_OUTCOME = {
    "Example A": "failed",
    "Example B": "alive",
    "Example C": "alive",
}

st.caption(
    "Three real companies from the untouched test data are available below, so you do not "
    "need to type in 18 figures by hand. Their true outcome is hidden until after you predict."
)
chosen_example = st.selectbox("Load an example company (optional):", list(EXAMPLE_COMPANIES.keys()))

if chosen_example != "-- Enter your own figures --" and st.session_state.get("_loaded_example") != chosen_example:
    for feat, val in EXAMPLE_COMPANIES[chosen_example].items():
        st.session_state[f"input_{feat}"] = val
    st.session_state["_loaded_example"] = chosen_example
    st.rerun()
elif chosen_example == "-- Enter your own figures --" and st.session_state.get("_loaded_example") is not None:
    # Clear previously loaded example values so switching back to manual
    # entry does not leave old figures behind, confusingly.
    for feat in all_feature_columns:
        st.session_state[f"input_{feat}"] = 0.0
    st.session_state["_loaded_example"] = None
    st.rerun()

user_values = {}

# Render the inputs grouped by financial category rather than as one
# undifferentiated block of 18 boxes. Each group gets its own heading, and
# fields within a group are laid out two-per-row.
for group_name, group_features in FEATURE_GROUPS.items():
    st.markdown(f"**{group_name}**")
    for i in range(0, len(group_features), 2):
        cols = st.columns(2)
        for j, feat in enumerate(group_features[i:i+2]):
            with cols[j]:
                label = f"{feat} - {FEATURE_LABELS.get(feat, '')}"
                user_values[feat] = st.number_input(
                    label, value=0.0, format="%.3f", key=f"input_{feat}"
                )
    st.write("")  # small gap between groups

if st.button("Predict Bankruptcy Risk", type="primary"):
    raw_input = np.array([[user_values[f] for f in all_feature_columns]])
    scaled_input = scaler.transform(raw_input)
    selected_idx = [all_feature_columns.index(f) for f in selected_features]
    model_input = scaled_input[:, selected_idx]

    rf_opinion = rf_model.predict_proba(model_input)[:, 1][0]
    gb_opinion = gb_model.predict_proba(model_input)[:, 1][0]
    knn_opinion = knn_model.predict_proba(model_input)[:, 1][0]

    combined_input = np.array([[rf_opinion, gb_opinion, knn_opinion]])
    final_probability = meta_model.predict_proba(combined_input)[:, 1][0]

    st.subheader("Result")
    risk_percent = final_probability * 100

    if final_probability >= chosen_threshold:
        st.error(f"Estimated bankruptcy risk: {risk_percent:.1f}% - HIGH RISK "
                  f"(above the {chosen_threshold:.0%} threshold)")
    else:
        st.success(f"Estimated bankruptcy risk: {risk_percent:.1f}% - LOWER RISK "
                   f"(below the {chosen_threshold:.0%} threshold)")

    
    # ADDITION 1: bar chart comparing all three experts plus the
    # final combined result, instead of plain text sentences only.
    
    st.write("### What each expert thought, compared to the final result")
    opinions_df = pd.DataFrame({
        "Model": ["Random Forest", "Gradient Boosting", "k-Nearest Neighbours", "FINAL (combined)"],
        "Risk (%)": [rf_opinion * 100, gb_opinion * 100, knn_opinion * 100, risk_percent],
    })

    fig_op, ax_op = plt.subplots(figsize=(6.5, 3.8))
    bar_colors = ["#CADCFC", "#CADCFC", "#CADCFC", "#C0392B"]
    ax_op.bar(opinions_df["Model"], opinions_df["Risk (%)"], color=bar_colors)
    ax_op.axhline(chosen_threshold * 100, color="#555555", linestyle="--", linewidth=1,
                  label=f"Chosen threshold ({chosen_threshold:.0%})")
    ax_op.set_ylabel("Estimated risk (%)")
    ax_op.set_title("Expert Opinions vs. Final Combined Result")
    ax_op.legend(fontsize=8)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    st.pyplot(fig_op)

    st.write(
        "The 'manager' model (meta-learner) weighs these three opinions "
        "together to produce the final result above, based on how "
        "reliable each expert was found to be during training - it does "
        "not simply average them."
    )

    # Reveal the real outcome only now, only for example companies, and
    # only after the participant has already seen the model's prediction.
    if chosen_example in EXAMPLE_TRUE_OUTCOME:
        true_outcome = EXAMPLE_TRUE_OUTCOME[chosen_example]
        st.write("---")
        if true_outcome == "failed":
            st.warning(f"**What actually happened:** this company genuinely went bankrupt "
                       f"in the real, untouched test data.")
        else:
            st.info(f"**What actually happened:** this company genuinely remained in "
                   f"business in the real, untouched test data.")

    st.caption(
        "Note: this prototype is trained on historical US company data "
        "(1999-2018) and is intended to demonstrate the research method, "
        "not to be used for real investment decisions."
    )

