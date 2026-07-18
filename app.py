import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import smtplib
from email.mime.text import MIMEText

# ---------------- EMAIL CONFIG ----------------
def send_email_alert(receiver_email, patient_name, age, condition, location):

    sender_email = "divyasriveeraragavan22@gmail.com"
    app_password = "uspe lzvb wahj hmsz"

    subject = "🚨 Emergency Alert"

    body = f"""
Emergency Alert!

Patient Name: {patient_name}
Age: {age}
Condition: {condition}
Location: {location}

Immediate medical attention is required.
"""

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender_email, app_password)
        server.send_message(msg)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="“ElderWatch AI: Smart Health Monitoring with Predictive Risk Analysis", layout="wide")

# ---------------- SESSION STATE ----------------
if "patient_name" not in st.session_state:
    st.session_state.patient_name = "Sarojini"
    st.session_state.age = 65
    st.session_state.condition = "Heart Risk"
    st.session_state.caretaker = "caretaker@gmail.com"
    st.session_state.location = "Chennai, India"
    st.session_state.alarm_active = False

if "health_history" not in st.session_state:
    st.session_state.health_history = []

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("## ⚙️ Settings")

st.session_state.patient_name = st.sidebar.text_input("Patient Name", st.session_state.patient_name)
st.session_state.age = st.sidebar.number_input("Age", 1, 120, st.session_state.age)
st.session_state.condition = st.sidebar.text_input(" Condition", st.session_state.condition)
st.session_state.caretaker = st.session_state.caretaker = st.sidebar.text_input(" Caretaker Email", st.session_state.caretaker)
st.session_state.location = st.sidebar.text_input(" Location", st.session_state.location)

menu = st.sidebar.radio(" Navigation", [" Home", " Monitor", " Analytics", " Emergency"])

# ---------------- TITLE ----------------
st.title(" “ElderWatch AI: Smart Health Monitoring with Predictive Risk Analysis")
st.write(f"📍 Location: {st.session_state.location}")

# ---------------- HOME ----------------
if menu == " Home":
    st.subheader(" Welcome")

    st.write(f"Welcome **{st.session_state.patient_name}** ")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", "92%", "+2%")
    col2.metric("Devices", "12", "+3")
    col3.metric("Alerts", "2", "-1")

    # NEW (ONLY ADDITION)
    st.markdown("###  System Overview")
    st.info("""
    This system monitors elderly health in real-time using AI.
    It analyzes vital signs, detects risks, and sends emergency alerts instantly.
    """)

    st.markdown("###  Features")
    st.write("""
    - Real-time monitoring  
    - Risk detection  
    - Emergency alert system  
    - SMS alerts using Twilio  
    """)


# ---------------- MONITOR ----------------
elif menu == " Monitor":
    st.subheader(" Live Patient Monitoring")

    col1, col2 = st.columns(2)

    heart_rate = col1.slider(" Heart Rate", 40, 180, 75)
    bp = col2.slider(" Blood Pressure", 80, 180, 120)

    oxygen = st.slider(" Oxygen Level (SpO2)", 70, 100, 98)
    temp = st.slider(" Temperature", 95, 105, 98)

    st.write(f"""
     Heart Rate: {heart_rate}  
     BP: {bp}  
     Oxygen: {oxygen}%  
     Temp: {temp}°F  
    """)

    # Save history (UPDATED with BP)
    st.session_state.health_history.append({
        "Heart Rate": heart_rate,
        "BP": bp,
        "Oxygen": oxygen,
        "Temp": temp
    })

    # Risk calculation
    risk_score = 0
    if heart_rate > 120: risk_score += 30
    if bp > 150: risk_score += 25
    if oxygen < 90: risk_score += 35
    if temp > 102: risk_score += 20

    st.markdown("### Risk Score")
    st.progress(min(risk_score / 100, 1.0))
    st.write(f"Risk Score: **{risk_score}/100**")

    #  CRITICAL CONDITION DISPLAY
    if risk_score > 70:
        st.error(" CRITICAL CONDITION – Immediate medical attention required!")
    elif risk_score > 40:
        st.warning(" Moderate Risk – Monitor closely")
    else:
        st.success("Stable Condition")

    #  AI SUGGESTIONS (FOCUS)
    st.markdown("### AI Health Suggestions")

    if oxygen < 90:
        st.error("Provide oxygen support immediately!")
    if heart_rate > 120:
        st.warning("Ensure patient rests and reduce activity.")
    if bp > 150:
        st.warning("High BP detected – avoid stress and monitor closely.")
    if temp > 102:
        st.warning("Possible fever – consult doctor.")

    if risk_score == 0:
        st.success("Patient is stable. Maintain regular monitoring.")

# ---------------- ANALYTICS ----------------
elif menu == " Analytics":
    st.subheader("Model Analytics")

    data = pd.DataFrame({
        "Round": [1,2,3,4,5],
        "Accuracy": [65,75,82,88,92],
        "Loss": [0.8,0.6,0.4,0.25,0.12],
        "Precision": [60,70,78,85,90]
    })

    col1, col2 = st.columns(2)

    # Accuracy
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data["Round"], y=data["Accuracy"], mode='lines+markers'))
    fig1.update_layout(title="Accuracy")
    col1.plotly_chart(fig1, use_container_width=True)

    # Loss
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=data["Round"], y=data["Loss"], mode='lines+markers'))
    fig2.update_layout(title="Loss")
    col2.plotly_chart(fig2, use_container_width=True)

    # Precision
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=data["Round"], y=data["Precision"], mode='lines+markers'))
    fig3.update_layout(title="Precision")
    st.plotly_chart(fig3, use_container_width=True)

    #  HEALTH TRENDS (FIXED)
    if len(st.session_state.health_history) > 1:
        df = pd.DataFrame(st.session_state.health_history)

        st.markdown("###  Health Trends")

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(y=df["Heart Rate"], mode='lines', name='Heart Rate'))
        fig4.add_trace(go.Scatter(y=df["BP"], mode='lines', name='Blood Pressure'))
        fig4.add_trace(go.Scatter(y=df["Oxygen"], mode='lines', name='Oxygen'))
        fig4.add_trace(go.Scatter(y=df["Temp"], mode='lines', name='Temperature'))

        fig4.update_layout(title="Health Trends Over Time")
        st.plotly_chart(fig4, use_container_width=True)

    # ---------------- HEALTH TRENDS ----------------
    if len(st.session_state.health_history) > 2:
        df = pd.DataFrame(st.session_state.health_history)

        st.markdown("###  Health Trends")

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(y=df["Heart Rate"], mode='lines', name='Heart Rate'))
        fig4.add_trace(go.Scatter(y=df["Oxygen"], mode='lines', name='Oxygen'))
        fig4.add_trace(go.Scatter(y=df["Temp"], mode='lines', name='Temperature'))

        fig4.update_layout(title="Health Trends Over Time")

        st.plotly_chart(fig4, use_container_width=True)
# ---------------- EMERGENCY ----------------
elif menu == " Emergency":
    st.subheader(" Emergency")

    if st.button(" Send Alert"):
        st.session_state.alarm_active = True

        send_email_alert(
            st.session_state.caretaker,
            st.session_state.patient_name,
            st.session_state.age,
            st.session_state.condition,
            st.session_state.location
        )

        st.success("✅ Emergency Email Sent Successfully!")

    if st.session_state.alarm_active:
        st.markdown("""
        <div style='text-align:center; font-size:40px; color:red; animation: blink 1s infinite;'>
        🚨 EMERGENCY IN PROGRESS 🚨
        </div>

        <style>
        @keyframes blink {
            0% { opacity: 1; }
            50% { opacity: 0.3; }
            100% { opacity: 1; }
        }
        </style>
        """, unsafe_allow_html=True)
# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(" Built with Streamlit") 
