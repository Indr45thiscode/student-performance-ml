# 🎓 Student Performance Prediction using Machine Learning

## 📌 Project Overview

This project implements a Machine Learning-based Student Performance Prediction System using regression techniques.
The objective of the system is to analyze academic factors such as study hours, attendance percentage, assignment scores, and previous test scores to predict a student’s final performance.
A realistic dataset of 500 records was generated using weighted relationships and controlled noise to simulate real-world academic patterns.

Two regression models were implemented and compared:

- Linear Regression (Baseline Model)
- Random Forest Regression (Improved Model)

Model performance was evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

In addition to model training and evaluation, the system includes a dynamic user input feature. Users can enter student details manually, and the trained model generates real-time predictions of the final score.

This project demonstrates a complete end-to-end machine learning workflow including data generation, preprocessing, model training, evaluation, model comparison, and real-time prediction.


## 🎯 Problem Statement
Educational institutions often need to evaluate student performance based on various academic factors.  
Manual evaluation can be inefficient and may not reveal hidden patterns.  
This project uses Machine Learning to predict final student scores based on academic inputs.

## 📊 Dataset Details
The dataset contains 500 records with the following features:

- Study_Hours
- Attendance_Percentage
- Assignment_Score
- Previous_Test_Score
- Final_Score (Target Variable)

The dataset was generated using a weighted formula with realistic noise to simulate real-world academic performance patterns.

## ⚙️ Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn


## 🧠 Machine Learning Models

### 1️⃣ Linear Regression
A basic regression model used to understand linear relationships between features and target.

### 2️⃣ Random Forest Regression
An ensemble learning method that improves prediction accuracy by combining multiple decision trees.


## 📈 Model Evaluation

Evaluation metrics used:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

The model with lower error values is considered better performing.

## 📊 Result Visualization
The project includes a scatter plot comparing:
- Actual Final Scores
- Predicted Final Scores

This visualization helps in understanding model accuracy.

## 🖥️ System Functionality

The notebook allows users to enter student details such as study hours, attendance percentage, assignment score, and previous test score.

The system then generates a predicted final score using the trained Random Forest model.

## How to Run the Project

1. Clone the repository:
    git clone https://github.com/Indr45thiscode/student-performance-ml.git

2. Install required libraries:
    pip install pandas numpy matplotlib scikit-learn

3. Run the Jupyter Notebook:
    jupyter notebook

4. Open `student_prediction.ipynb` and execute all cells.


## 📌 Conclusion
The project demonstrates a complete machine learning workflow including data generation, model training, evaluation, comparison, and real-time user-based prediction.


## 🔮 Future Improvements
- Use real-world educational datasets
- Add more performance-related features
- Deploy the model with a simple UI
- Hyperparameter tuning for improved accuracy


## 👨‍💻 Author
Indrajit Sawant
