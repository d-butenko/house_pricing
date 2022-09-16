
import pandas as pd

import mlflow
import mlflow.pyfunc


import alibi
from alibi_detect.cd import TabularDrift

import constants
from preprocessing import Preprocessing
from learning import Learning


class ModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
        super().__init__()

class Main:
    '''Performs ML pipeline operations'''

    def __init__(self):
        self.data_learn_path = constants.DATA_LEARN_PATH
        self.data_predict_path = constants.DATA_PREDICT_PATH
        self.output_path = constants.OUTPUT_PATH

        self.data_learn = None
        self.data_predict = None

    def load_data(self, path=None):
        '''Loads data from csv to DataFrame'''     
        data = pd.read_csv(path)
        return data

    def save_result(self, data=None, path=None):
        '''Saves results into csv'''
        data.to_csv(path, index=False)

    def drift_check(self):

        learning = Learning()
        X_ref, X_test, _, _ = learning.train_test_split(data=self.data_learn, target='SalePrice', test_size=0.5)
        cd = TabularDrift(x_ref=X_ref.values, p_val=.05)
        preds = cd.predict(1.1*X_test.values)
        
        return preds['data']['is_drift']

    def main(self):
        '''Core method that runs through all the pipeline steps'''
        # load data
        self.data_learn = self.load_data(self.data_learn_path)
        self.data_predict = self.load_data(self.data_predict_path)
        
        # preprocess
        preprocessing = Preprocessing(self.data_learn, self.data_predict)
        self.data_learn, self.data_predict = preprocessing.preprocess()

        # drift check
        print('Drift result')
        drift_result = self.drift_check()
        print(drift_result)

        # train and predict
        learning = Learning()
        X_train, X_test, y_train, y_test = learning.train_test_split(data=self.data_learn, target='SalePrice', test_size=0.25)

        with mlflow.start_run():
            fitted_estimator = learning.train_model(features=X_train, labels=y_train)       
            predictions = learning.make_prediction(estimator=fitted_estimator, data_predict=self.data_predict)
            score = learning.get_score(estimator=fitted_estimator, X_test=X_test, y_test=y_test)

            mlflow.log_metric('R2', score)

            mlflow.pyfunc.log_model('model', python_model=ModelWrapper(fitted_estimator))

        # save results
        result = pd.DataFrame({'Id': preprocessing.predict_ids, 'SalePrice': predictions})
        save_path = self.output_path
        self.save_result(result, save_path)

        return result


class_instance = Main()
print(class_instance.main())
