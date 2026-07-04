import logging
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logger = logging.getLogger(__name__)

def get_feature_lists(df: pd.DataFrame, target_col: str = 'SalePrice', id_col: str = 'Id'):
    features_df = df.drop(columns=[col for col in [id_col, target_col] if col in df.columns], errors='ignore')
    numeric_features = features_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = features_df.select_dtypes(include=['object', 'category']).columns.tolist()
    logger.info(f"Identified {len(numeric_features)} numerical features and {len(categorical_features)} categorical features.")
    return numeric_features, categorical_features

def build_preprocessing_pipeline(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    logger.info("Building ColumnTransformer preprocessing pipeline...")
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    return preprocessor

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    initial_rows = df.shape[0]
    subset_cols = [c for c in df.columns if c != 'Id']
    cleaned_df = df.drop_duplicates(subset=subset_cols)
    removed = initial_rows - cleaned_df.shape[0]
    if removed > 0:
        logger.info(f"Removed {removed} duplicate rows.")
    else:
        logger.info("No duplicate rows found.")
    return cleaned_df
