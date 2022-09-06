
import pandas as pd

import constants
from preprocessing import Preprocessing
from learning import Learning

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

    def main(self):
        '''Core method that runs through all the pipeline steps'''

        # load data
        self.data_learn = self.load_data(self.data_learn_path)
        self.data_predict = self.load_data(self.data_predict_path)
        
        # preprocess
        preprocessing = Preprocessing(self.data_learn, self.data_predict)
        self.data_learn, self.data_predict = preprocessing.preprocess()

        # train and predict
        learning = Learning()
        X_train, _, y_train, _ = learning.train_test_split(data=self.data_learn, target='SalePrice', 
                                                            test_size=0.25)

        fitted_estimator = learning.train_model(features=X_train, labels=y_train)       
        predictions = learning.make_prediction(estimator=fitted_estimator, data_predict=self.data_predict)

        # save results
        result = pd.DataFrame({'Id': preprocessing.predict_ids, 'SalePrice': predictions})
        save_path = self.output_path
        self.save_result(result, save_path)

        return result


class_instance = Main()
print(class_instance.main())
