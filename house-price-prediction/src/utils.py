import os
import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories():
    directories = ['data', 'src', 'models', 'outputs']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
        else:
            logger.info(f"Directory already exists: {directory}")

def generate_synthetic_data(train_path: str, test_path: str, num_train_samples: int = 200, num_test_samples: int = 100):
    logger.info("Generating synthetic Kaggle-like datasets...")
    np.random.seed(42)

    numeric_configs = {
        'MSSubClass': lambda n: np.random.choice([20, 30, 45, 50, 60, 70, 75, 80, 85, 90, 120, 160, 180, 190], size=n),
        'LotFrontage': lambda n: np.random.normal(70, 20, size=n).clip(20, 200),
        'LotArea': lambda n: np.random.normal(10000, 4000, size=n).clip(1300, 50000),
        'OverallQual': lambda n: np.random.randint(1, 11, size=n),
        'OverallCond': lambda n: np.random.randint(1, 11, size=n),
        'YearBuilt': lambda n: np.random.randint(1950, 2021, size=n),
        'YearRemodAdd': lambda n: np.random.randint(1950, 2021, size=n),
        'MasVnrArea': lambda n: np.random.choice([0.0, 100.0, 200.0, 300.0], size=n, p=[0.6, 0.2, 0.1, 0.1]),
        'BsmtFinSF1': lambda n: np.random.normal(400, 300, size=n).clip(0, 2000),
        'BsmtFinSF2': lambda n: np.random.choice([0.0, 50.0, 100.0], size=n, p=[0.9, 0.07, 0.03]),
        'BsmtUnfSF': lambda n: np.random.normal(500, 300, size=n).clip(0, 1500),
        'TotalBsmtSF': lambda n: np.random.normal(1000, 400, size=n).clip(0, 3000),
        '1stFlrSF': lambda n: np.random.normal(1100, 350, size=n).clip(300, 3000),
        '2ndFlrSF': lambda n: np.random.choice([0, 500, 800, 1000], size=n, p=[0.5, 0.1, 0.2, 0.2]),
        'LowQualFinSF': lambda n: np.random.choice([0, 50, 100], size=n, p=[0.98, 0.01, 0.01]),
        'GrLivArea': lambda n: np.random.normal(1500, 500, size=n).clip(300, 4500),
        'BsmtFullBath': lambda n: np.random.choice([0, 1, 2], size=n, p=[0.5, 0.45, 0.05]),
        'BsmtHalfBath': lambda n: np.random.choice([0, 1], size=n, p=[0.95, 0.05]),
        'FullBath': lambda n: np.random.choice([1, 2, 3], size=n, p=[0.3, 0.6, 0.1]),
        'HalfBath': lambda n: np.random.choice([0, 1, 2], size=n, p=[0.5, 0.48, 0.02]),
        'BedroomAbvGr': lambda n: np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.05, 0.2, 0.55, 0.17, 0.03]),
        'KitchenAbvGr': lambda n: np.random.choice([1, 2], size=n, p=[0.98, 0.02]),
        'TotRmsAbvGrd': lambda n: np.random.randint(4, 12, size=n),
        'Fireplaces': lambda n: np.random.choice([0, 1, 2, 3], size=n, p=[0.5, 0.4, 0.08, 0.02]),
        'GarageYrBlt': lambda n: np.random.randint(1950, 2021, size=n),
        'GarageCars': lambda n: np.random.choice([0, 1, 2, 3, 4], size=n, p=[0.05, 0.25, 0.55, 0.14, 0.01]),
        'GarageArea': lambda n: np.random.normal(450, 180, size=n).clip(0, 1200),
        'WoodDeckSF': lambda n: np.random.normal(100, 100, size=n).clip(0, 800),
        'OpenPorchSF': lambda n: np.random.normal(50, 50, size=n).clip(0, 500),
        'EnclosedPorch': lambda n: np.random.normal(20, 30, size=n).clip(0, 300),
        '3SsnPorch': lambda n: np.random.choice([0, 100], size=n, p=[0.99, 0.01]),
        'ScreenPorch': lambda n: np.random.normal(15, 25, size=n).clip(0, 300),
        'PoolArea': lambda n: np.random.choice([0, 500], size=n, p=[0.995, 0.005]),
        'MiscVal': lambda n: np.random.choice([0, 400, 700], size=n, p=[0.98, 0.01, 0.01]),
        'MoSold': lambda n: np.random.randint(1, 13, size=n),
        'YrSold': lambda n: np.random.randint(2006, 2011, size=n),
    }

    categorical_configs = {
        'MSZoning': ['RL', 'RM', 'C (all)', 'FV', 'RH'],
        'Street': ['Pave', 'Grvl'],
        'Alley': ['Pave', 'Grvl', np.nan],
        'LotShape': ['Reg', 'IR1', 'IR2', 'IR3'],
        'LandContour': ['Lvl', 'Bnk', 'HLS', 'Low'],
        'Utilities': ['AllPub', 'NoSeWa'],
        'LotConfig': ['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3'],
        'LandSlope': ['Gtl', 'Mod', 'Sev'],
        'Neighborhood': ['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst', 'NWAmes', 'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes', 'SawyerW', 'IDOTRR', 'MeadowV', 'Edwards', 'Timber', 'Gilbert', 'StoneBr', 'ClearCr', 'NPkVill', 'Blmngtn', 'BrDale', 'SWISU', 'Blueste'],
        'Condition1': ['Norm', 'Feedr', 'Artery', 'RRAn', 'PosN', 'RRAe', 'PosA', 'RRNn', 'RRNe'],
        'Condition2': ['Norm', 'Feedr', 'Artery', 'RRNn', 'PosN', 'PosA', 'RRAn', 'RRAe'],
        'BldgType': ['1Fam', 'TwnhsE', 'Twnhs', 'Duplex', '2fmCon'],
        'HouseStyle': ['2Story', '1Story', '1.5Fin', '1.5Unf', 'SFoyer', 'SLvl', '2.5Unf', '2.5Fin'],
        'RoofStyle': ['Gable', 'Hip', 'Gambrel', 'Mansard', 'Flat', 'Shed'],
        'RoofMatl': ['CompShg', 'WdShngl', 'WdShake', 'Tar&Grv', 'Metal', 'ClyTile', 'Membran', 'Roll'],
        'Exterior1st': ['VinylSd', 'MetalSd', 'Wd Sdng', 'HdBoard', 'BrkFace', 'WdShing', 'CemntBd', 'Plywood', 'AsbShng', 'Stucco', 'BrkComm', 'AsphShn', 'Stone', 'ImStucc', 'CBlock'],
        'Exterior2nd': ['VinylSd', 'MetalSd', 'Wd Shng', 'HdBoard', 'Plywood', 'Wd Sdng', 'CmentBd', 'BrkFace', 'Stucco', 'AsbShng', 'Brk Cmn', 'ImStucc', 'AsphShn', 'Stone', 'Other', 'CBlock'],
        'MasVnrType': ['BrkFace', 'None', 'Stone', 'BrkCmn', np.nan],
        'ExterQual': ['Gd', 'TA', 'Ex', 'Fa'],
        'ExterCond': ['TA', 'Gd', 'Fa', 'Po', 'Ex'],
        'Foundation': ['PConc', 'CBlock', 'BrkTil', 'Wood', 'Slab', 'Stone'],
        'BsmtQual': ['Gd', 'TA', 'Ex', np.nan, 'Fa'],
        'BsmtCond': ['TA', 'Gd', np.nan, 'Fa', 'Po'],
        'BsmtExposure': ['No', 'Gd', 'Mn', 'Av', np.nan],
        'BsmtFinType1': ['GLQ', 'ALQ', 'Unf', 'Rec', 'BLQ', np.nan, 'LwQ'],
        'BsmtFinType2': ['Unf', 'BLQ', np.nan, 'ALQ', 'Rec', 'LwQ', 'GLQ'],
        'Heating': ['GasA', 'GasW', 'Grav', 'Wall', 'OthW', 'Floor'],
        'HeatingQC': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
        'CentralAir': ['Y', 'N'],
        'Electrical': ['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix', np.nan],
        'KitchenQual': ['Gd', 'TA', 'Ex', 'Fa'],
        'Functional': ['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev'],
        'FireplaceQu': [np.nan, 'TA', 'Gd', 'Fa', 'Ex', 'Po'],
        'GarageType': ['Attchd', 'Detchd', 'BuiltIn', 'CarPort', np.nan, 'Basment', '2Types'],
        'GarageFinish': ['RFn', 'Unf', 'Fin', np.nan],
        'GarageQual': ['TA', 'Fa', 'Gd', np.nan, 'Ex', 'Po'],
        'GarageCond': ['TA', 'Fa', np.nan, 'Gd', 'Po', 'Ex'],
        'PavedDrive': ['Y', 'N', 'P'],
        'PoolQC': [np.nan, 'Ex', 'Fa', 'Gd'],
        'Fence': [np.nan, 'MnPrv', 'GdWo', 'GdPrv', 'MnWw'],
        'MiscFeature': [np.nan, 'Shed', 'Gar2', 'Othr', 'TenC'],
        'SaleType': ['WD', 'New', 'COD', 'ConLD', 'ConLI', 'CWD', 'ConLw', 'Con', 'Oth'],
        'SaleCondition': ['Normal', 'Abnorml', 'Partial', 'AdjLand', 'Alloca', 'Family']
    }

    def make_dataset(n: int, is_train: bool):
        data = {}
        data['Id'] = np.arange(1 if is_train else num_train_samples + 1, (num_train_samples + 1 if is_train else num_train_samples + num_test_samples + 1))
        
        for col, generator in numeric_configs.items():
            data[col] = generator(n)
            if col in ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']:
                mask = np.random.choice([True, False], size=n, p=[0.05, 0.95])
                data[col] = np.where(mask, np.nan, data[col])

        for col, categories in categorical_configs.items():
            cleaned_cats = [c if isinstance(c, str) else 'Missing' for c in categories]
            choices = np.random.choice(cleaned_cats, size=n)
            data[col] = [np.nan if x == 'Missing' else x for x in choices]

        df = pd.DataFrame(data)

        if is_train:
            qual = df['OverallQual'].fillna(5)
            liv_area = df['GrLivArea'].fillna(1500)
            yr_built = df['YearBuilt'].fillna(1980)
            cars = df['GarageCars'].fillna(2)
            
            base_price = 50000
            price = (
                base_price 
                + qual * 20000 
                + liv_area * 60 
                + (yr_built - 1950) * 150 
                + cars * 12000 
                + np.random.normal(0, 15000, size=n)
            )
            df['SalePrice'] = price.clip(30000, 800000)

        return df

    train_df = make_dataset(num_train_samples, is_train=True)
    test_df = make_dataset(num_test_samples, is_train=False)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info(f"Synthetic training data saved to {train_path} (Shape: {train_df.shape})")
    logger.info(f"Synthetic test data saved to {test_path} (Shape: {test_df.shape})")
