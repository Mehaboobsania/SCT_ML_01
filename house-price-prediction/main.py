import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import setup_directories, generate_synthetic_data
from src.eda import run_basic_eda
from src.preprocessing import clean_data, get_feature_lists, build_preprocessing_pipeline
from src.model import (
    train_linear_regression,
    evaluate_model,
    plot_regression_diagnostics,
    save_trained_model,
    predict_houses
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting House Price Prediction pipeline...")
    
    setup_directories()
    
    train_path = os.path.join('data', 'train.csv')
    test_path = os.path.join('data', 'test.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        logger.info("Kaggle house price dataset train.csv or test.csv not found in data/ directory.")
        generate_synthetic_data(train_path, test_path, num_train_samples=400, num_test_samples=100)
    else:
        logger.info("Found Kaggle house price datasets in data/ directory.")

    logger.info(f"Loading training data from {train_path}...")
    train_df = pd.read_csv(train_path)
    
    logger.info("Running Exploratory Data Analysis...")
    run_basic_eda(train_df, output_dir='outputs')
    
    logger.info("Cleaning data and building preprocessing pipeline...")
    train_df = clean_data(train_df)
    
    if 'SalePrice' not in train_df.columns:
        raise ValueError("Target column 'SalePrice' is missing from the training dataset.")
        
    X = train_df.drop(columns=['Id', 'SalePrice'], errors='ignore')
    y = train_df['SalePrice']
    
    numeric_features, categorical_features = get_feature_lists(train_df, target_col='SalePrice', id_col='Id')
    
    preprocessor = build_preprocessing_pipeline(numeric_features, categorical_features)
    
    logger.info("Splitting dataset into 80% training and 20% validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Training split shape: {X_train.shape}, Validation split shape: {X_val.shape}")
    
    logger.info("Training Linear Regression model...")
    model_pipeline = train_linear_regression(X_train, y_train, preprocessor)
    
    logger.info("Evaluating model metrics...")
    metrics, y_val_pred = evaluate_model(model_pipeline, X_val, y_val)
    
    logger.info("Generating evaluation diagnostic plots...")
    plot_regression_diagnostics(y_val, y_val_pred, output_dir='outputs')
    
    model_save_path = os.path.join('models', 'linear_regression_model.joblib')
    save_trained_model(model_pipeline, model_save_path)
    
    logger.info(f"Loading test data from {test_path}...")
    test_df = pd.read_csv(test_path)
    
    test_ids = test_df['Id'] if 'Id' in test_df.columns else pd.Series(range(len(test_df)))
    test_features = test_df.drop(columns=['Id', 'SalePrice'], errors='ignore')
    
    logger.info("Generating predictions on the test dataset...")
    test_preds = predict_houses(model_pipeline, test_features)
    
    submission_df = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': test_preds
    })
    submission_path = os.path.join('outputs', 'predictions_submission.csv')
    submission_df.to_csv(submission_path, index=False)
    logger.info(f"Test set predictions saved to {submission_path} (Shape: {submission_df.shape})")
    
    logger.info("Demonstrating prediction logic on a single new house sample...")
    
    sample_house = test_features.iloc[[0]].copy()
    sample_id = test_ids.iloc[0]
    
    pred_price = predict_houses(model_pipeline, sample_house)[0]
    logger.info("=" * 60)
    logger.info(f"Prediction for House ID {sample_id}:")
    logger.info(f" - GrLivArea (Living Area): {sample_house['GrLivArea'].values[0]:.1f} sqft")
    logger.info(f" - OverallQual (Overall Quality Score): {sample_house['OverallQual'].values[0]}/10")
    logger.info(f" - YearBuilt (Construction Year): {int(sample_house['YearBuilt'].values[0])}")
    logger.info(f" - Predicted Sale Price: ${pred_price:,.2f}")
    logger.info("=" * 60)
    
    logger.info("House Price Prediction pipeline successfully completed.")

if __name__ == '__main__':
    main()
