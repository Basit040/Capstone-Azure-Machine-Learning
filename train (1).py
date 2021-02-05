# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 19:56:52 2021
@author: Abdul Basit
"""

import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory


    
    
data = "https://raw.githubusercontent.com/Basit040/Capstone-Azure-Machine-Learning/main/titanicdataset.csv"
#creating data in Tabular format via TabularDatasetFactory
ds = TabularDatasetFactory.from_delimited_files(data)
run = Run.get_context()

# clean data function
# Extracting x and y from clean data function
x= ds.to_pandas_dataframe()
y=x.pop('Survived')

# Split data into train and test sets.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()
    
    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/hyperDrive_{}_{}'.format(args.C,args.max_iter))

if __name__ == '__main__':
    main()

