import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# ---------------- LOAD DATA ----------------
data = pd.read_csv("dataset/student_data.csv")

features = [
    "study_hours",
    "attendance",
    "previous_score",
    "assignments_completed"
]

X = data[features]
y = data["final_score"]

# ---------------- TRAIN MODEL ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# ---------------- MODEL EVALUATION ----------------
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# ---------------- TITLE ----------------
st.title("🎓 Student Performance Predictor")

st.write(
    "Predict a student's final academic score using "
    "study hours, attendance, previous score, and assignments completed."
)

st.divider()

# ---------------- INPUT SECTION ----------------
st.subheader("📋 Enter Student Details")

study_hours = st.number_input(
    "📚 Study Hours per Day",
    min_value=0.0,
    max_value=24.0,
    value=6.0,
    step=0.5
)

attendance = st.number_input(
    "📅 Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=85.0,
    step=1.0
)

previous_score = st.number_input(
    "📝 Previous Score",
    min_value=0.0,
    max_value=100.0,
    value=75.0,
    step=1.0
)

assignments_completed = st.number_input(
    "✅ Assignments Completed",
    min_value=0,
    max_value=10,
    value=8,
    step=1
)

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Performance", use_container_width=True):

    student = pd.DataFrame(
        [[
            study_hours,
            attendance,
            previous_score,
            assignments_completed
        ]],
        columns=features
    )

    prediction = model.predict(student)[0]

    prediction = max(0, min(100, prediction))

    st.success(
        f"### 🎯 Predicted Final Score: {prediction:.2f}"
    )

    if prediction >= 80:
        st.info("🌟 Excellent predicted performance!")
    elif prediction >= 60:
        st.info("👍 Good predicted performance.")
    else:
        st.warning("📚 More improvement may be needed.")

# ---------------- MODEL PERFORMANCE ----------------
st.divider()

st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("Mean Absolute Error", f"{mae:.2f}")

with col2:
    st.metric("R² Score", f"{r2:.2f}")

# ---------------- ABOUT ----------------
st.divider()

st.subheader("ℹ️ About This Project")

st.write(
    "This project uses Linear Regression to predict student "
    "final scores from academic and study-related features."
)

st.caption(
    "Built with Python, Pandas, Scikit-learn and Streamlit."
)

st.caption("Developed by Juber Khan")