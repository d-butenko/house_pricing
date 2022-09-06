import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import constants

class Preprocessing:
    '''Handles preprocessing steps'''
    def __init__(self, data_learn=None, data_predict=None):
        self.data_learn = data_learn
        self.data_predict = data_predict
        self.data = None
        self.predict_ids = None

    def concat_data(self, data_learn=None, data_predict=None, target='SalePrice'):
        '''Concatanates learning data and data for prediction'''
        data = pd.concat([data_learn.drop(target, axis=1), data_predict])
        data.reset_index(drop=True, inplace=True)

        return data

    def separate_data(self, target='SalePrice'):
        '''Separates concatanated data'''
        data_learn = self.data[:self.data_learn.shape[0]]
        data_predict = self.data[self.data_learn.shape[0]:]

        data_learn[target] = self.data_learn[target]

        return data_learn, data_predict
        
    def log_transform(self, data=None, columns=None):
        '''Performs log(1+x) transformatiom'''
        for column in columns:
            data[column] = np.log1p(data[column])

        return data

    def fill_nans(self):
        '''Fills missing values'''
        # NaN -> 'None'
        for column in constants.FEATURES_NAN2CONST:
            self.data[column].fillna(value='None', inplace=True)
        
        # NaN -> 0
        self.data['MasVnrArea'].fillna(value=0, inplace=True)

        # NaN -> 0, not NaN -> 1
        for column in constants.FEATURES_NAN2BINARY:
            self.data['temp'] = 0
            self.data.loc[self.data[column].notnull(), 'temp'] = 1
            self.data[column] = self.data['temp']
            
        self.data.drop(['temp'], axis=1, inplace=True)

        # filling according to values distribution in corresponding feature
        for column in self.data.columns:
            value_distr = self.data[column].value_counts(normalize=True)
            nans = self.data[column].isnull()
            self.data.loc[nans, column] = np.random.choice(value_distr.index, 
                                                            size=len(self.data[nans]), 
                                                            p=value_distr.values)

    def remove_outliers(self, data=None, columns=None):
        '''Removes outliers'''
        for column in columns:
            # Find the mean and standard dev
            std = data[column].std()
            mean = data[column].mean()

            # Calculate the cutoff
            cut_off = std * 3
            lower, upper = mean - cut_off, mean + cut_off

            # Trim the outliers
            data = data[(data[column] < upper) & (data[column] > lower)]
        
        return data

    def redefine_categories(self):
        '''Handles poor populated categories'''
        ex2gd = ['ExterQual', 'ExterCond', 'BsmtQual', 'KitchenQual', 'FireplaceQu']

        for column in ex2gd:
            self.data.loc[self.data[column]=='Ex', column] = 'Gd'

        for feature, categories in constants.FEATURES_TO_COMBINE.items():
            self.data.loc[-self.data[feature].isin(categories), feature] = 'Other'

    def scale_numerical(self, columns=None):
        '''Normalizes numerical features'''
        scaler = StandardScaler()
        for column in columns:
            scaler.fit(self.data[[column]])
            self.data[column] = scaler.transform(self.data[[column]])

    def preprocess(self):
        '''Core preprocessing method'''
        # store ids from prediction dataset
        self.predict_ids = self.data_predict['Id']

        # combine data for learning with data for prediction to perform preprocessing together
        self.data = self.concat_data(data_learn=self.data_learn, data_predict=self.data_predict, 
                                    target='SalePrice')

        # new feature
        self.data['Age'] = self.data['YrSold'] - self.data['YearBuilt']

        # features that will not be used for analysis
        self.data.drop(constants.FEATURES_TO_DROP, axis=1, inplace=True)

        # handle missing values
        self.fill_nans()

        # handle poor populated categories
        self.redefine_categories()

        # normalize numerical features
        self.scale_numerical(columns=constants.FEATURES_TO_SCALE)

        # encode categorical features
        self.data = pd.get_dummies(self.data, columns=constants.FEATURES_TO_ENCODE)    

        # split back into learning data and data for prediction
        data_learn, data_predict = self.separate_data(target='SalePrice')
       
        return data_learn, data_predict