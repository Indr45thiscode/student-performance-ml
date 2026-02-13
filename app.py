import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------
# Generate Dataset
# ---------------------------
rows = 500

study_hours = np.random.randint(1, 11, rows)
attendance = np.random.randint(50, 101, rows)
assignment_score = np.random.randint(40, 101, rows)
previous_test_score = np.random.randint(40, 101, rows)

final_score = (
    0.4 * study_hours * 10 +
    0.3 * attendance +
    0.2 * assignment_score +
    0.1 * previous_test_score +
    np.random.normal(0, 5, rows)
)

data = pd.DataFrame({
    'Study_Hours': study_hours,
    'Attendance_Percentage': attendance,
    'Assignment_Score': assignment_score,
    'Previous_Test_Score': previous_test_score,
    'Final_Score': final_score
})

X = data[['Study_Hours', 'Attendance_Percentage',
          'Assignment_Score', 'Previous_Test_Score']]
y = data['Final_Score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Train Models
# ---------------------------
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluation
y_pred_lin = lin_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

mae_lin = mean_absolute_error(y_test, y_pred_lin)
rmse_lin = np.sqrt(mean_squared_error(y_test, y_pred_lin))

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🎓 Student Performance Prediction Dashboard")

st.sidebar.header("📊 Model Selection")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Random Forest", "Linear Regression", "Compare Both"]
)

st.subheader("Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    study = st.slider("Study Hours", 1, 10, 5)
    attendance = st.slider("Attendance Percentage", 50, 100, 75)

with col2:
    assignment = st.slider("Assignment Score", 40, 100, 70)
    previous = st.slider("Previous Test Score", 40, 100, 70)

new_student = pd.DataFrame({
    'Study_Hours': [study],
    'Attendance_Percentage': [attendance],
    'Assignment_Score': [assignment],
    'Previous_Test_Score': [previous]
})

if st.button("Predict Final Score"):

    if model_choice == "Random Forest":
        prediction = rf_model.predict(new_student)
        st.success(f"Random Forest Prediction: {round(prediction[0], 2)}")

    elif model_choice == "Linear Regression":
        prediction = lin_model.predict(new_student)
        st.success(f"Linear Regression Prediction: {round(prediction[0], 2)}")

    else:
        pred_rf = rf_model.predict(new_student)
        pred_lin = lin_model.predict(new_student)

        st.success(f"Random Forest Prediction: {round(pred_rf[0], 2)}")
        st.info(f"Linear Regression Prediction: {round(pred_lin[0], 2)}")

st.markdown("---")
st.subheader("📈 Model Performance")

col3, col4 = st.columns(2)

with col3:
    st.write("**Linear Regression**")
    st.write(f"MAE: {round(mae_lin, 2)}")
    st.write(f"RMSE: {round(rmse_lin, 2)}")

with col4:
    st.write("**Random Forest**")
    st.write(f"MAE: {round(mae_rf, 2)}")
    st.write(f"RMSE: {round(rmse_rf, 2)}")

# Feature Importance
st.markdown("---")
st.subheader("📌 Feature Importance (Random Forest)")

importance = rf_model.feature_importances_
st.write("Total Importance:", round(sum(importance), 4))

feature_names = X.columns

fig, ax = plt.subplots()
ax.barh(feature_names, importance)
ax.set_xlabel("Importance")
ax.set_title("Feature Importance")
st.pyplot(fig)
