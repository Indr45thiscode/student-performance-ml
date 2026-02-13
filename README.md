# 🎓 Student Performance Prediction using Machine Learning

## 📌 Project Overview
This project focuses on predicting student final scores using Machine Learning techniques.  
The objective is to analyze student-related features and build regression models to estimate performance.

Two regression models were implemented and compared:
- Linear Regression
- Random Forest Regression

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

## How to Run the Project

1. Clone the repository:
    git clone https://github.com/Indr45thiscode/student-performance-ml.git

2. Install required libraries:
    pip install pandas numpy matplotlib scikit-learn

3. Run the Jupyter Notebook:
    jupyter notebook

4. Open `student_prediction.ipynb` and execute all cells.


## 📌 Conclusion
The project successfully demonstrates the implementation of regression models for predicting student performance.  
Model comparison shows the effectiveness of ensemble methods like Random Forest over basic regression.


## 🔮 Future Improvements
- Use real-world educational datasets
- Add more performance-related features
- Deploy the model with a simple UI
- Hyperparameter tuning for improved accuracy


## 👨‍💻 Author
Indrajit Sawant
