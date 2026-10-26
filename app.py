import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

st.set_page_config(page_title="Campus Facility Dashboard", page_icon="🏫", layout="wide")

st.title("🏫 Campus Facility Reservation Analysis")
st.caption("Reservation monitoring, exploratory analysis, decision-tree classification, and risk ratings.")
st.info("This app automatically creates the sample dataset from the original notebook. No CSV upload is required. Risk ratings are model predictions and must not be used as the sole basis for rejecting reservations.")

@st.cache_data
def make_data():
    data = {
        "Reservation_ID": [f"R{i:03d}" for i in range(1, 16)],
        "Facility_Type": ["Computer Lab", "Gymnasium", "Library", "Computer Lab", "Auditorium", "Gymnasium", "Library", "Computer Lab", "Auditorium", "Gymnasium", "Library", "Computer Lab", "Auditorium", "Gymnasium", "Library"],
        "User_Category": ["Student", "Faculty", "Student", "Faculty", "Organization", "Student", "Faculty", "Student", "Organization", "Faculty", "Student", "Organization", "Faculty", "Student", "Faculty"],
        "Lead_Time_Days": [2, 10, 5, 20, 15, 1, 7, 12, 3, 18, 9, 4, 14, 6, 21],
        "Participants": [25, 100, 15, 30, 200, 80, 20, 35, 150, 90, 18, 40, 180, 75, 22],
        "Actual_Attendance": [23, 95, 15, 0, 180, 0, 18, 30, 140, 85, 16, 0, 170, 70, 20],
        "Usage_Status": ["Used", "Used", "Used", "Cancelled", "Used", "No-Show", "Used", "Used", "Used", "Used", "Used", "Cancelled", "Used", "Used", "Used"]
    }
    return pd.DataFrame(data)

df = make_data()
raw_df = df.copy()
raw_df.loc[0, "Facility_Type"] = " computer lab "
raw_df.loc[1, "User_Category"] = "faculty"
raw_df.loc[2, "Participants"] = -5
raw_df.loc[3, "Lead_Time_Days"] = -2
raw_df.loc[4, "Participants"] = np.nan
raw_df.loc[5, "Facility_Type"] = np.nan
raw_df = pd.concat([raw_df, raw_df.iloc[[0]]], ignore_index=True)
clean_df = raw_df.drop_duplicates().copy()
for col in ["Facility_Type", "User_Category", "Usage_Status"]:
    clean_df[col] = clean_df[col].apply(lambda x: x.strip().title() if isinstance(x, str) else x)
for col in ["Facility_Type", "User_Category"]:
    clean_df[col] = clean_df[col].fillna("Unknown")
for col in ["Lead_Time_Days", "Participants", "Actual_Attendance"]:
    clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
clean_df.loc[clean_df["Lead_Time_Days"] < 0, "Lead_Time_Days"] = np.nan
clean_df.loc[clean_df["Participants"] < 0, "Participants"] = np.nan
clean_df.loc[clean_df["Actual_Attendance"] < 0, "Actual_Attendance"] = np.nan
invalid_attendance = clean_df["Actual_Attendance"].notna() & clean_df["Participants"].notna() & (clean_df["Actual_Attendance"] > clean_df["Participants"])
clean_df.loc[invalid_attendance, "Actual_Attendance"] = np.nan
conditions = [clean_df["Lead_Time_Days"].between(0, 2), clean_df["Lead_Time_Days"].between(3, 7), clean_df["Lead_Time_Days"].between(8, 14), clean_df["Lead_Time_Days"] > 14]
clean_df["Lead_Time_Category"] = np.select(conditions, ["Last Minute", "Short Lead", "Medium Lead", "Early Booking"], default="Unknown")
clean_df.loc[clean_df["Lead_Time_Days"].isna(), "Lead_Time_Category"] = np.nan
status_map = {"Used": "Completed", "Cancelled": "Not Completed", "No-Show": "Not Completed"}
clean_df["Reservation_Outcome"] = clean_df["Usage_Status"].map(status_map)
denominator_valid = clean_df["Participants"].notna() & (clean_df["Participants"] > 0)
clean_df["Attendance_Ratio"] = np.nan
clean_df.loc[denominator_valid, "Attendance_Ratio"] = clean_df.loc[denominator_valid, "Actual_Attendance"] / clean_df.loc[denominator_valid, "Participants"]

# Fit a reproducible model using the same predictors and target concept as the notebook.
model_df = clean_df[["Facility_Type", "User_Category", "Lead_Time_Days", "Participants", "Reservation_Outcome"]].dropna(subset=["Reservation_Outcome"]).copy()
for col in ["Facility_Type", "User_Category"]:
    model_df[col] = model_df[col].fillna("Unknown").astype(str)
for col in ["Lead_Time_Days", "Participants"]:
    model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    model_df[col] = model_df[col].fillna(model_df[col].median() if model_df[col].notna().any() else 0)
model_df = model_df[model_df["Reservation_Outcome"].isin(["Completed", "Not Completed"])]
categorical = ["Facility_Type", "User_Category"]
numerical = ["Lead_Time_Days", "Participants"]
predictor_columns = categorical + numerical
model = None
metrics = None
if model_df["Reservation_Outcome"].nunique() >= 2 and model_df["Reservation_Outcome"].value_counts().min() >= 2:
    X, y = model_df[predictor_columns], model_df["Reservation_Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    preprocessor = ColumnTransformer([("categorical", OneHotEncoder(handle_unknown="ignore"), categorical), ("numerical", "passthrough", numerical)])
    model = Pipeline([("preprocessor", preprocessor), ("classifier", DecisionTreeClassifier(random_state=42, class_weight="balanced", max_depth=4))])
    model.fit(X_train, y_train)
    pred_test = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, pred_test),
        "Precision": precision_score(y_test, pred_test, pos_label="Not Completed", zero_division=0),
        "Recall": recall_score(y_test, pred_test, pos_label="Not Completed", zero_division=0),
        "F1-score": f1_score(y_test, pred_test, pos_label="Not Completed", zero_division=0),
        "Confusion matrix": confusion_matrix(y_test, pred_test, labels=["Completed", "Not Completed"])
    }

dashboard_df = clean_df.copy()
status = dashboard_df["Usage_Status"].astype(str).str.strip().str.lower()
total = len(dashboard_df)
completed = int((status == "used").sum())
cancelled = int((status == "cancelled").sum())
no_show = int(status.isin(["no-show", "no show"]).sum())
dashboard_df["Predicted Outcome"] = model.predict(dashboard_df[predictor_columns]) if model is not None else "Unavailable"
if model is not None and hasattr(model, "predict_proba") and "Not Completed" in model.classes_:
    prob = model.predict_proba(dashboard_df[predictor_columns])[:, list(model.classes_).index("Not Completed")] * 100
    ratings = np.where(prob >= 70, "HIGH", np.where(prob >= 40, "MEDIUM", "LOW"))
else:
    prob = np.full(len(dashboard_df), np.nan)
    ratings = np.where(dashboard_df["Predicted Outcome"] == "Not Completed", "HIGH", "LOW")
dashboard_df["Risk Score (%)"] = np.round(prob, 1)
dashboard_df["Risk Rating"] = ratings

tabs = st.tabs(["Overview", "Dataset & Cleaning", "Exploratory Analysis", "Decision Tree", "Requirements", "Risk Dashboard", "Critical Analysis"])

with tabs[0]:
    st.subheader("Reservation Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reservations", total)
    c2.metric("Completed / Used", completed)
    c3.metric("Cancelled", cancelled)
    c4.metric("No-show", no_show)
    c5, c6 = st.columns(2)
    c5.metric("Completion Rate", f"{completed / total * 100:.1f}%" if total else "0%")
    c6.metric("Cancellation Rate", f"{cancelled / total * 100:.1f}%" if total else "0%")
    st.subheader("Dataset Preview")
    st.dataframe(clean_df.head(10), use_container_width=True)
    st.download_button("Download cleaned dataset (CSV)", clean_df.to_csv(index=False).encode("utf-8"), "cleaned_facility_reservations.csv", "text/csv")

with tabs[1]:
    st.subheader("Dataset Acquisition")
    st.write("A 15-record sample facility reservation dataset is created automatically when the app runs.")
    st.dataframe(df, use_container_width=True)
    st.write("**Original data dimensions:**", f"{df.shape[0]} rows × {df.shape[1]} columns")
    st.subheader("Data Quality Before Cleaning")
    st.write(f"Rows before cleaning: {len(raw_df)}")
    st.write(f"Duplicate rows before cleaning: {int(raw_df.duplicated().sum())}")
    st.dataframe(raw_df.isnull().sum().rename("Missing Values").to_frame(), use_container_width=True)
    st.subheader("Cleaned Dataset")
    st.write(f"Rows after cleaning: {len(clean_df)}")
    st.dataframe(clean_df, use_container_width=True)
    st.subheader("Cleaning and Feature Engineering Applied")
    st.markdown("- Removed exact duplicate rows.\n- Standardized category/status capitalization and whitespace.\n- Replaced missing facility/user categories with `Unknown`.\n- Converted numeric fields and marked negative values invalid.\n- Removed impossible attendance values where attendance exceeds participants.\n- Created `Lead_Time_Category`, `Reservation_Outcome`, and `Attendance_Ratio`.")
    st.download_button("Download original dataset (CSV)", df.to_csv(index=False).encode("utf-8"), "facility_reservations_original.csv", "text/csv")

with tabs[2]:
    st.subheader("Exploratory Data Analysis")
    left, right = st.columns(2)
    facility_counts = clean_df["Facility_Type"].value_counts()
    with left:
        st.markdown("**Reservation Count by Facility Type**")
        st.bar_chart(facility_counts)
        if not facility_counts.empty:
            st.write(f"Highest reservation count: **{facility_counts.idxmax()}** ({facility_counts.max()} reservations).")
    with right:
        st.markdown("**Cancellation / No-show by User Category**")
        statuses = clean_df["Usage_Status"].astype(str).str.strip().str.lower()
        bad = statuses.isin(["cancelled", "no-show", "no show"])
        user_summary = clean_df.assign(Cancelled_No_Show=bad).groupby("User_Category").agg(Total_Reservations=("Reservation_ID", "count"), Cancelled_No_Show=("Cancelled_No_Show", "sum"))
        user_summary["Cancellation_No_Show_Percentage"] = (user_summary["Cancelled_No_Show"] / user_summary["Total_Reservations"].replace(0, np.nan) * 100).round(2)
        st.dataframe(user_summary, use_container_width=True)
        st.bar_chart(user_summary["Cancellation_No_Show_Percentage"])
    st.markdown("**Lead Time Category vs Reservation Outcome**")
    outcome_table = pd.crosstab(clean_df["Lead_Time_Category"], clean_df["Reservation_Outcome"])
    st.dataframe(outcome_table, use_container_width=True)
    st.bar_chart(outcome_table)

with tabs[3]:
    st.subheader("Decision Tree Classification")
    if model is None:
        st.warning("The sample does not contain enough examples per class to train a stratified decision tree.")
    else:
        cols = st.columns(4)
        for col, (label, val) in zip(cols, [(k, v) for k, v in metrics.items() if k != "Confusion matrix"]):
            col.metric(label, f"{val:.3f}")
        st.caption("Evaluation uses an 80/20 train-test split and random_state=42. The sample is small, so metrics may change substantially with new data.")
        st.markdown("**Confusion Matrix** (rows = actual; columns = predicted; order: Completed, Not Completed)")
        cm = pd.DataFrame(metrics["Confusion matrix"], index=["Actual Completed", "Actual Not Completed"], columns=["Predicted Completed", "Predicted Not Completed"])
        st.dataframe(cm, use_container_width=True)
        st.markdown("**Top Predictor Importance**")
        pre = model.named_steps["preprocessor"]
        names = pre.get_feature_names_out()
        importances = model.named_steps["classifier"].feature_importances_
        imp = pd.DataFrame({"Feature": names, "Importance": importances}).sort_values("Importance", ascending=False)
        st.dataframe(imp.head(10), use_container_width=True)
        fig, ax = plt.subplots(figsize=(16, 8))
        plot_tree(model.named_steps["classifier"], feature_names=names, class_names=model.named_steps["classifier"].classes_, filled=True, rounded=True, max_depth=3, fontsize=8, ax=ax)
        ax.set_title("Decision Tree Classifier")
        st.pyplot(fig)
        plt.close(fig)

with tabs[4]:
    st.subheader("Part V — Functional Requirements")
    requirements = [
        {"Finding / Evidence": "Part III: Reservation counts were analyzed by facility type.", "Requirement ID": "FR-DM-01", "Functional Requirement": "The system shall generate a reservation summary showing the number of reservations for each facility type to help administrators monitor facility demand."},
        {"Finding / Evidence": "Part III: Cancellation and no-show percentages were compared across user categories.", "Requirement ID": "FR-DM-02", "Functional Requirement": "The system shall display cancellation and no-show counts and percentages by user category to help administrators monitor reservation patterns."},
        {"Finding / Evidence": "Part IV: A Decision Tree model was trained to predict reservation outcomes.", "Requirement ID": "FR-DM-03", "Functional Requirement": "The system shall display the predicted reservation outcome and model evaluation results to support administrator decision-making."}
    ]
    req_df = pd.DataFrame(requirements)
    st.dataframe(req_df, use_container_width=True)
    st.download_button("Download functional requirements (CSV)", req_df.to_csv(index=False).encode("utf-8"), "task9_functional_requirements.csv", "text/csv")

with tabs[5]:
    st.subheader("Facility Administrator Risk Dashboard")
    st.caption("Observed status and predicted risk are different: a rating is an estimate from the model, not a confirmed outcome.")
    high = int((dashboard_df["Risk Rating"] == "HIGH").sum())
    medium = int((dashboard_df["Risk Rating"] == "MEDIUM").sum())
    low = int((dashboard_df["Risk Rating"] == "LOW").sum())
    a,b,c = st.columns(3)
    a.metric("High Risk", high)
    b.metric("Medium Risk", medium)
    c.metric("Low Risk", low)
    st.markdown("**Risk Rating Rules**")
    st.write("HIGH: risk score ≥ 70%; MEDIUM: 40–69.9%; LOW: below 40%. If probabilities are unavailable, the app uses the predicted outcome to assign a basic rating.")
    filter_rating = st.multiselect("Filter by risk rating", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM", "LOW"])
    filtered = dashboard_df[dashboard_df["Risk Rating"].isin(filter_rating)]
    st.dataframe(filtered[["Reservation_ID", "Facility_Type", "User_Category", "Usage_Status", "Predicted Outcome", "Risk Score (%)", "Risk Rating"]], use_container_width=True)
    st.download_button("Download risk predictions (CSV)", dashboard_df[["Reservation_ID", "Facility_Type", "User_Category", "Usage_Status", "Predicted Outcome", "Risk Score (%)", "Risk Rating"]].to_csv(index=False).encode("utf-8"), "reservation_risk_predictions.csv", "text/csv")
    st.bar_chart(dashboard_df["Risk Rating"].value_counts())
    st.warning("Do not automatically reject a reservation because it is rated HIGH risk. Review the details and use human judgment.")

with tabs[6]:
    st.subheader("Part VI — Task 12: Critical Analysis")
    st.markdown("""
    **A. Why should predicted high risk not automatically reject a reservation?**  
    A predicted high risk should not automatically reject a reservation because the model can make incorrect predictions. The administrator should review the reservation details before making a decision. Predictions should support human judgment, not replace it.

    **B. How does poor data quality affect the model and information system?**  
    Poor data quality, such as missing values, duplicate records, and incorrect information, can produce unreliable predictions and misleading reports. The system should validate reservation data and maintain accurate records to improve information quality.

    **C. What information should be displayed to explain a high-risk prediction?**  
    The dashboard should show the predicted risk level, the factors that contributed to the prediction, the reservation details used, and a clear notice that the result is only a prediction. The administrator should review this information before making a final decision.
    """)
