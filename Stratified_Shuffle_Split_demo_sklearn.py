#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 11:59:44 2018

@author: densonsmith
"""

import pandas as pd

import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit


df_input = pd.read_csv('SSS_demo_data.csv')


df_input_headers = df_input.columns.tolist()

y_header = df_input_headers[0]

X_headers = df_input_headers[1:]

sss = StratifiedShuffleSplit(n_splits=1,
                             test_size=0.2,
                             random_state=42)





for train_index,test_index in sss.split(df_input[X_headers],df_input[y_header]):
    X_train = df_input.loc[train_index, X_headers] 
    y_train = df_input.loc[train_index, y_header]

    X_test = df_input.loc[test_index, X_headers] 
    y_test = df_input.loc[test_index, y_header]
    
    print(train_index[0:5])
    print(X_train.head())