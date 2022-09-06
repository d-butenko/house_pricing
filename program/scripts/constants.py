import os

WORK_DIR = os.getcwd()

DATA_LEARN_PATH = WORK_DIR + '\\data\\train.csv'
DATA_PREDICT_PATH = WORK_DIR + '\\data\\test.csv'
OUTPUT_PATH = WORK_DIR + '\\data\\output.csv'

FEATURES_TO_DROP = ['Id', 'Street', 'Alley', 'LandContour', 'LandSlope', 'Condition1', 'Condition2', 'RoofMatl', 'Heating', 'Electrical', 
                    'SaleType', 'SaleCondition', 'Utilities', 'BsmtCond', 'BsmtFinType2', 'Functional', 'GarageQual', 'GarageCond', 
                    'PavedDrive', 'CentralAir', 'PoolQC', 'MiscFeature', 'MSSubClass', 'BsmtHalfBath', 'KitchenAbvGr', 'BsmtFinSF2', 
                    'LowQualFinSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'YrSold', 'YearBuilt', 
                    'YearRemodAdd', 'GarageYrBlt', 'LotFrontage', 'GarageArea', 'GrLivArea', 'TotalBsmtSF', 'Exterior2nd']

FEATURES_NAN2CONST = ['GarageFinish', 'FireplaceQu', 'BsmtFinType1', 'BsmtExposure', 'BsmtQual', 'GarageType']

FEATURES_TO_COMBINE = {'MSZoning': ['RL', 'RM', 'FV'],
                        'LotShape': ['Reg', 'IR1'],
                        'LotConfig': ['Inside', 'Corner', 'CulDSac'],
                        'Neighborhood': ['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst', 
                                        'NWAmes', 'Gilbert', 'NridgHt', 'Sawyer', 'BrkSide', 
                                        'SawyerW', 'Crawfor', 'Mitchel', 'NoRidge', 'Timber', 
                                        'IDOTRR'],
                        'HouseStyle': ['1Story', '2Story', '1.5Fin', 'SLvl'],
                        'RoofStyle': ['Gable', 'Hip'],
                        'Exterior1st': ['VinylSd', 'HdBoard', 'MetalSd', 'Wd Sdng', 'Plywood', 
                                        'CemntBd', 'BrkFace', 'Stucco', 'WdShing', 'AsbShng'],
                        'MasVnrType': ['None', 'BrkFace', 'Stone'],
                        'Foundation': ['PConc', 'CBlock', 'BrkTil'],
                        'GarageType': ['Attchd', 'Detchd', 'BuiltIn', 'None'],
                        'ExterQual': ['TA', 'Gd'],
                        'ExterCond': ['TA', 'Gd'],
                        'KitchenQual': ['TA', 'Gd'],
                        'BsmtQual': ['TA', 'Gd', 'None'],
                        'FireplaceQu': ['TA', 'Gd', 'None'],
                        'HeatingQC': ['Ex', 'Gd', 'TA']}

FEATURES_NAN2BINARY = ['Fence']

FEATURES_TO_SCALE = ['LotArea', 'OverallQual', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF', 
                    'BsmtFullBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'WoodDeckSF', 
                    'OpenPorchSF', 'Fence', 'Age']

FEATURES_TO_ENCODE = ['MSZoning', 'LotShape', 'LotConfig', 'Neighborhood', 'BldgType', 'HouseStyle', 'RoofStyle', 'Exterior1st',
                    'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'HeatingQC',
                    'KitchenQual', 'FireplaceQu', 'GarageType', 'GarageFinish', 'MoSold']
