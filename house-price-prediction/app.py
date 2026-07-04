import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="House Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    h1 {
        color: #f0f2f6;
        font-family: 'Outfit', sans-serif;
    }
    .stNumberInput > label {
        color: #f0f2f6;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏡 House Price Prediction Dashboard")
st.write("Enter house parameters to estimate the sale price using our trained machine learning pipeline.")

model_path = os.path.join('models', 'linear_regression_model.joblib')
if not os.path.exists(model_path):
    st.error("Trained model pipeline not found. Please run the training script first.")
    st.stop()

model = joblib.load(model_path)

st.sidebar.header("Configure House Features")

overall_qual = st.sidebar.slider("Overall Quality (1-10)", 1, 10, 6)
gr_liv_area = st.sidebar.number_input("Above Grade Living Area (sqft)", min_value=300, max_value=6000, value=1500, step=50)
year_built = st.sidebar.slider("Year Built", 1900, 2026, 2000)
garage_cars = st.sidebar.slider("Garage Car Capacity", 0, 4, 2)
lot_area = st.sidebar.number_input("Lot Area (sqft)", min_value=500, max_value=100000, value=9000, step=500)
full_bath = st.sidebar.slider("Full Bathrooms", 1, 4, 2)

input_data = pd.DataFrame([{
    'MSSubClass': 60,
    'LotFrontage': 70.0,
    'LotArea': float(lot_area),
    'OverallQual': int(overall_qual),
    'OverallCond': 5,
    'YearBuilt': int(year_built),
    'YearRemodAdd': int(year_built),
    'MasVnrArea': 0.0,
    'BsmtFinSF1': 400.0,
    'BsmtFinSF2': 0.0,
    'BsmtUnfSF': 500.0,
    'TotalBsmtSF': 900.0,
    '1stFlrSF': 1000.0,
    '2ndFlrSF': 800.0,
    'LowQualFinSF': 0.0,
    'GrLivArea': float(gr_liv_area),
    'BsmtFullBath': 1,
    'BsmtHalfBath': 0,
    'FullBath': int(full_bath),
    'HalfBath': 1,
    'BedroomAbvGr': 3,
    'KitchenAbvGr': 1,
    'TotRmsAbvGrd': 7,
    'Fireplaces': 1,
    'GarageYrBlt': int(year_built),
    'GarageCars': int(garage_cars),
    'GarageArea': float(garage_cars * 200),
    'WoodDeckSF': 0.0,
    'OpenPorchSF': 0.0,
    'EnclosedPorch': 0.0,
    '3SsnPorch': 0.0,
    'ScreenPorch': 0.0,
    'PoolArea': 0.0,
    'MiscVal': 0.0,
    'MoSold': 6,
    'YrSold': 2010,
    'MSZoning': 'RL',
    'Street': 'Pave',
    'Alley': np.nan,
    'LotShape': 'Reg',
    'LandContour': 'Lvl',
    'Utilities': 'AllPub',
    'LotConfig': 'Inside',
    'LandSlope': 'Gtl',
    'Neighborhood': 'CollgCr',
    'Condition1': 'Norm',
    'Condition2': 'Norm',
    'BldgType': '1Fam',
    'HouseStyle': '2Story',
    'RoofStyle': 'Gable',
    'RoofMatl': 'CompShg',
    'Exterior1st': 'VinylSd',
    'Exterior2nd': 'VinylSd',
    'MasVnrType': 'None',
    'ExterQual': 'TA',
    'ExterCond': 'TA',
    'Foundation': 'PConc',
    'BsmtQual': 'TA',
    'BsmtCond': 'TA',
    'BsmtExposure': 'No',
    'BsmtFinType1': 'Unf',
    'BsmtFinType2': 'Unf',
    'Heating': 'GasA',
    'HeatingQC': 'Ex',
    'CentralAir': 'Y',
    'Electrical': 'SBrkr',
    'KitchenQual': 'TA',
    'Functional': 'Typ',
    'FireplaceQu': np.nan,
    'GarageType': 'Attchd',
    'GarageFinish': 'RFn',
    'GarageQual': 'TA',
    'GarageCond': 'TA',
    'PavedDrive': 'Y',
    'PoolQC': np.nan,
    'Fence': np.nan,
    'MiscFeature': np.nan,
    'SaleType': 'WD',
    'SaleCondition': 'Normal'
}])

prediction = model.predict(input_data)[0]
prediction = max(0, prediction)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Key Parameters")
    metrics_data = {
        "Feature": ["Overall Quality", "Living Area (sqft)", "Year Built", "Garage Cars", "Lot Area"],
        "Selected Value": [f"{overall_qual}/10", f"{gr_liv_area} sqft", year_built, garage_cars, f"{lot_area} sqft"]
    }
    st.table(pd.DataFrame(metrics_data))

with col2:
    st.subheader("Predicted Sale Price")
    st.markdown(f"<h1 style='color: #2ecc71; font-size: 3rem;'>${prediction:,.2f}</h1>", unsafe_allow_html=True)
    st.metric(label="Est. Value", value=f"${prediction:,.2f}")

st.markdown("---")
st.subheader("Model Diagnostic Context")

train_path = os.path.join('data', 'train.csv')
if os.path.exists(train_path):
    train_df = pd.read_csv(train_path)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(train_df['SalePrice'], kde=True, color='#2980b9', ax=ax)
    ax.axvline(prediction, color='#e74c3c', linestyle='--', linewidth=2, label=f'Your House (${prediction:,.0f})')
    ax.set_title("Predicted Price vs. Historical Price Distribution")
    ax.set_xlabel("Sale Price ($)")
    ax.legend()
    st.pyplot(fig)
