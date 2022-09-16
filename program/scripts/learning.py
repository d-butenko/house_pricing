import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error

# import mlflow
# import mlflow.pyfunc

class Learning:
    '''Handles learning and prediction process'''
    def __init__(self):
        pass

    def train_test_split(self, data=None, target=None, test_size=0.25):
        '''Splits data into train and test datasets'''
        y = data[target]
        X = data.drop([target], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                            random_state=42)
        
        return X_train, X_test, y_train, y_test

    def train_model(self, features=None, labels=None):
        '''Fits custom ML algorythm'''
        lasso = Lasso(alpha=0.001)
        ridge = Ridge(alpha=10)
        svm = LinearSVR(C=1)
        grb = GradientBoostingRegressor(learning_rate=0.05, max_depth=3, n_estimators=500, random_state=42)
        base_estimators = [('Lasso', lasso), ('Ridge', ridge), ('Gradient Boosting', grb)]
        stacking_ensemble = StackingRegressor(estimators=base_estimators, final_estimator=svm)

        stacking_ensemble.fit(features, labels)
            
        return stacking_ensemble

    def get_score(self, estimator=None, X_test=None, y_test=None):
        '''Returns R^2 score'''
        r2 = estimator.score(X_test, y_test)

        return r2

    def make_prediction(self, estimator=None, data_predict=None):
        '''Predicts unknown targets'''
        predictions = estimator.predict(data_predict)

        return predictions
