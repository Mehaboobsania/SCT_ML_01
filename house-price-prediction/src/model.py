import os
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series, preprocessor) -> Pipeline:
    logger.info("Initializing and training Linear Regression model...")
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    model_pipeline.fit(X_train, y_train)
    logger.info("Model training complete.")
    return model_pipeline

def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    logger.info("Evaluating model performance on test set...")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }
    logger.info(f"Evaluation Metrics:")
    logger.info(f" - Mean Absolute Error (MAE): ${mae:,.2f}")
    logger.info(f" - Mean Squared Error (MSE): {mse:,.2f}")
    logger.info(f" - Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    logger.info(f" - R² Score: {r2:.4f}")
    return metrics, y_pred

def plot_regression_diagnostics(y_true: pd.Series, y_pred: np.ndarray, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.7, color='#2980b9')
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], '--', color='#e74c3c', linewidth=2, label='Perfect Fit')
    plt.title('Actual vs. Predicted House Prices')
    plt.xlabel('Actual Sale Price ($)')
    plt.ylabel('Predicted Sale Price ($)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    actual_vs_pred_path = os.path.join(output_dir, 'actual_vs_predicted.png')
    plt.savefig(actual_vs_pred_path, dpi=150)
    plt.close()
    logger.info(f"Saved actual vs predicted plot to {actual_vs_pred_path}")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.7, color='#8e44ad')
    plt.axhline(y=0, color='#e74c3c', linestyle='--', linewidth=2)
    plt.title('Residual Plot')
    plt.xlabel('Predicted Sale Price ($)')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    residual_path = os.path.join(output_dir, 'residual_path.png')
    plt.savefig(residual_path, dpi=150)
    plt.close()
    logger.info(f"Saved residual plot to {residual_path}")

def save_trained_model(model: Pipeline, file_path: str):
    model_dir = os.path.dirname(file_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    joblib.dump(model, file_path)
    logger.info(f"Successfully saved model to: {file_path}")

def load_trained_model(file_path: str) -> Pipeline:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found at: {file_path}")
    model = joblib.load(file_path)
    logger.info(f"Successfully loaded model from: {file_path}")
    return model

def predict_houses(model: Pipeline, new_data: pd.DataFrame) -> np.ndarray:
    predictions = model.predict(new_data)
    predictions = np.clip(predictions, 0, None)
    return predictions
