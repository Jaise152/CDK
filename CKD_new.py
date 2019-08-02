# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 15:41:01 2019

@author: 611916967
"""

import numpy as np
import pandas as pd
#from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Imputer
from sklearn_pandas import DataFrameMapper, CategoricalImputer
#from sklearn.impute import SimpleImputer

#class Categorical_Imputer:
#    """
#    Imputing categorical data using the most frequent value
#    """
#    
#    # instance attribute
#    def __init__(self, strategy):
#        self.strategy = strategy
#        
#    # instance method
#    def fit_transform(self, df:'dataframe')->'dataframe':
#        """
#        Fill in missing categorical values using most frequent value
#        """
#        
#        # instantiate CategoricalImputer
#        imputer = CategoricalImputer()
#        
#        # convert array to dataframe
#        df_filled = df.apply(lambda x: imputer.fit_transform(x), axis=0)
#        
#        # return filled dataframe
#        return df_filled
#        

df = pd.read_csv('C:\Personal\MTech\Books and Materials\Data Mining\Assignment\Assignment_BLR (1)\kidneyChronic.csv')
df.replace('?', np.nan, inplace=True)
#
numerical_columns = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']
categorical_columns = df.columns.drop('class').drop(numerical_columns)


# convert numerical data 
df[numerical_columns] = df[numerical_columns].apply(pd.to_numeric, errors='coerce')
#
# check the number of unique values
df[categorical_columns].apply(lambda x: x.nunique(), axis=0)
#df['dm'].unique()
#df['cad'].unique()

df['dm']=df['dm'].str.strip()
df['cad']=df['cad'].str.strip()

df['class'] = df['class'].apply(lambda x: 1 if x=='ckd' else 0)

# define numerical imputer
num_imp = Imputer(missing_values=np.nan, strategy='median', axis=0)
# imputing on numerical data
df[numerical_columns] = num_imp.fit_transform(df[numerical_columns])

# define categorical imputer
cate_imputer = Categorical_Imputer('most_frequent')
# imputing on categorical data
df[categorical_columns] = cate_imputer.fit_transform(df[categorical_columns])
#
df[['htn','dm','cad','pe','ane']] = df[['htn','dm','cad','pe','ane']].replace(to_replace={'yes':1,'no':0})
df[['rbc','pc']] = df[['rbc','pc']].replace(to_replace={'abnormal':1,'normal':0})
df[['pcc','ba']] = df[['pcc','ba']].replace(to_replace={'present':1,'notpresent':0})
df[['appet']] = df[['appet']].replace(to_replace={'good':1,'poor':0})

df.corr().to_csv('C:\Personal\MTech\Books and Materials\Data Mining\Assignment\Assignment_BLR (1)\kidneyChronic_processed.csv')