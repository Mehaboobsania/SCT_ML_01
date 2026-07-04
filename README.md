# 🏠 House Price Prediction using Linear Regression

A Machine Learning project that predicts house prices based on various property features using the Linear Regression algorithm. This project demonstrates the complete machine learning workflow, including data preprocessing, exploratory data analysis (EDA), model training, evaluation, and prediction.

---

## 📌 Project Overview

The objective of this project is to build a regression model capable of estimating the sale price of a house based on its characteristics. The model is trained using the House Prices dataset from Kaggle and evaluated using standard regression metrics.

---

## 📂 Dataset

**Dataset:** House Prices: Advanced Regression Techniques

**Source:** Kaggle

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

The dataset contains various features describing residential homes, such as:

- Lot Area
- Overall Quality
- Overall Condition
- Year Built
- Total Basement Area
- Number of Bedrooms
- Number of Bathrooms
- Garage Area
- Living Area
- Sale Price (Target Variable)

---

## 🎯 Objectives

- Load and explore the dataset
- Clean and preprocess the data
- Handle missing values
- Encode categorical features
- Train a Linear Regression model
- Evaluate model performance
- Predict house prices for unseen data
- Save the trained model

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

---

## 📁 Project Structure

```
House-Price-Prediction/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── models/
│   └── linear_regression_model.pkl
│
├── outputs/
│   ├── correlation_heatmap.png
│   ├── actual_vs_predicted.png
│   └── residual_plot.png
│
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/your-username/House-Price-Prediction.git
```

Move into the project folder

```bash
cd House-Price-Prediction
```

Install the required packages

```bash
pip install -r requirements.txt
```

Download the Kaggle dataset and place the following files inside the **data/** folder:

- train.csv
- test.csv

---

## ▶️ Running the Project

Execute the main script

```bash
python main.py
```

The program will:

- Load the dataset
- Perform preprocessing
- Train the Linear Regression model
- Evaluate the model
- Display visualizations
- Save the trained model

---

## 📊 Model Evaluation

The model is evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## 📈 Visualizations

The project generates several visualizations including:

- Correlation Heatmap
- Feature Distribution
- Actual vs Predicted Prices
- Residual Plot

---

## 💾 Model Saving

The trained model is saved using Joblib for future predictions.

Example:

```python
import joblib

model = joblib.load("models/linear_regression_model.pkl")
```

---

## 🚀 Future Improvements

- Feature Engineering
- Hyperparameter Tuning
- Cross Validation
- Compare with Ridge and Lasso Regression
- Deploy as a Flask or Streamlit Web Application

---


## 👨‍💻 Author

**Shaik Mehaboob Sania**

B.Tech CSE | Machine Learning Enthusiast

GitHub: https://github.com/Mehaboobsania
LinkedIn: https://linkedin.com/in/mehaboob-sania-shaik-1b51142ba

---

## 📜 License

This project is developed for educational and internship purposes.
