import streamlit as st
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import joblib
import os

# -------------------------------------------------
# PAGE CONFIG (MUST BE FIRST STREAMLIT CALL)
# -------------------------------------------------
st.set_page_config(
    page_title="Human Activity Prediction",
    layout="centered"
)

# -------------------------------------------------
# PREMIUM UI (CUSTOM CSS)
# -------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}
h1, h2, h3 {
    color: #f8f9fa;
}
.stButton>button {
    background: linear-gradient(90deg, #ff512f, #dd2476);
    color: white;
    border-radius: 12px;
    height: 3em;
    font-size: 16px;
    border: none;
}
div[data-testid="stAlert"] {
    border-radius: 12px;
    font-size: 16px;
}
thead tr th {
    background-color: #1f4068;
    color: white;
}
tbody tr td {
    background-color: #162447;
    color: white;
}
.stProgress > div > div {
    background-image: linear-gradient(to right, #00f260, #0575e6);
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.title("🏃 Human Activity Prediction (Federated Learning)")
st.write("A premium URL-based prediction system using federated learning on sensor data")

# -------------------------------------------------
# SIDEBAR INFO
# -------------------------------------------------
st.sidebar.markdown("## 🧠 Model Information")
st.sidebar.write("**Model:** Federated Logistic Regression")
st.sidebar.write("**Clients:** 5")
st.sidebar.write("**Dataset:** UCI HAR Dataset")
st.sidebar.write("**Global Accuracy:** 94.74%")
st.sidebar.write("**Prediction Type:** Multiclass Activity Recognition")

# -------------------------------------------------
# LOAD DATA & MODEL
# -------------------------------------------------
@st.cache_data
def load_data():
    X_test = np.load("results/X_test_scaled.npy")
    y_test = pd.read_csv("results/y_test.csv")
    activity_labels = pd.read_csv(
        "data/UCI HAR Dataset/activity_labels.txt",
        sep=r"\s+",
        header=None,
        names=["id", "activity"]
    )
    return X_test, y_test, activity_labels

@st.cache_resource
def load_model():
    return joblib.load("results/global_model.pkl")

X_test_scaled, y_test, activity_labels = load_data()
model = load_model()
activities = activity_labels["activity"].tolist()

# -------------------------------------------------
# SESSION STATE (PREDICTION HISTORY)
# -------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------------------------
# PREDICTION BUTTON
# -------------------------------------------------
if st.button("🔮 Predict Random Activity"):
    i = random.randint(0, len(X_test_scaled) - 1)

    probs = model.predict_proba(X_test_scaled[i].reshape(1, -1))[0]
    pred_id = model.predict(X_test_scaled[i].reshape(1, -1))[0]

    actual = activity_labels.loc[
        activity_labels.id == y_test["label"].iloc[i], "activity"
    ].values[0]

    predicted = activity_labels.loc[
        activity_labels.id == pred_id, "activity"
    ].values[0]

    confidence = float(np.max(probs))

    # -------------------------------------------------
    # PREDICTION RESULT
    # -------------------------------------------------
    st.markdown("## 📌 Prediction Result")
    if predicted == actual:
        st.success(f"✅ Correct Prediction: **{predicted}**")
    else:
        st.error("❌ Incorrect Prediction")

    st.markdown(
        f"""
        **Actual Activity:** `{actual}`  
        **Predicted Activity:** `{predicted}`
        """
    )

    # -------------------------------------------------
    # CONFIDENCE BAR
    # -------------------------------------------------
    st.markdown("## 🎯 Prediction Confidence")
    st.progress(confidence)
    st.write(f"Model Confidence: **{confidence:.2%}**")

    # -------------------------------------------------
    # CONFIDENCE DONUT
    # -------------------------------------------------
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(
        [confidence, 1 - confidence],
        colors=["#00f260", "#333333"],
        startangle=90,
        wedgeprops=dict(width=0.3)
    )
    ax.text(0, 0, f"{confidence*100:.1f}%", ha="center", va="center",
            fontsize=18, color="white")
    ax.set_title("Confidence Gauge", color="white")
    st.pyplot(fig)

    # -------------------------------------------------
    # TOP-3 PREDICTIONS
    # -------------------------------------------------
    top_indices = np.argsort(probs)[-3:][::-1]
    top_df = pd.DataFrame({
        "Activity": [activities[j] for j in top_indices],
        "Confidence": [probs[j] for j in top_indices]
    })

    st.markdown("## 🏆 Top-3 Predicted Activities")
    st.table(top_df.style.format({"Confidence": "{:.2%}"}))

    # -------------------------------------------------
    # PROBABILITY BAR CHART
    # -------------------------------------------------
    st.markdown("## 📊 Prediction Probability Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(activities, probs)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # -------------------------------------------------
    # RADAR CHART (WOW)
    # -------------------------------------------------
    angles = np.linspace(0, 2 * np.pi, len(activities), endpoint=False)
    probs_radar = np.concatenate((probs, [probs[0]]))
    angles = np.concatenate((angles, [angles[0]]))

    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, probs_radar, color="#00f260", linewidth=2)
    ax.fill(angles, probs_radar, color="#00f260", alpha=0.25)
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, activities)
    ax.set_title("Activity Confidence Radar", color="white", pad=20)
    ax.tick_params(colors="white")
    st.pyplot(fig)

    # -------------------------------------------------
    # MISCLASSIFICATION INSIGHT
    # -------------------------------------------------
    st.markdown("## 🔍 Prediction Insight")
    if predicted != actual:
        st.warning(
            f"The model confused **{actual}** with **{predicted}** due to similarity "
            "in sensor patterns between static activities."
        )
    else:
        st.info(
            "The model shows strong confidence with clear separation between activity classes."
        )

    # -------------------------------------------------
    # SAVE HISTORY
    # -------------------------------------------------
    st.session_state.history.append({
        "Sample ID": i,
        "Actual": actual,
        "Predicted": predicted,
        "Confidence": round(confidence, 3)
    })

# -------------------------------------------------
# HISTORY DASHBOARD
# -------------------------------------------------
if len(st.session_state.history) > 0:
    st.markdown("## 📜 Prediction History (Session)")
    st.dataframe(pd.DataFrame(st.session_state.history))

# -------------------------------------------------
# FUTURE SCOPE
# -------------------------------------------------
st.markdown("## 🚀 Future Scope")
st.markdown("""
- 🔹 Real-time wearable sensor integration  
- 🔹 Edge-device federated learning  
- 🔹 Smart healthcare activity monitoring  
- 🔹 Privacy-preserving fitness tracking  
- 🔹 IoT-based human behavior analytics  
""")
