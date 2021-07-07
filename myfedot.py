
import os 

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Plots
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 18, 7

import warnings
warnings.filterwarnings('ignore')

# Prerocessing for FEDOT
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

# FEDOT 
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.chains.chain import Chain

def simple_autoregression_function(dataframe):
    """
    The function trains a linear autoregressive model, 
    which use 2 predictors to give a forecast
    
    :param dataframe: pandas DataFrame with time series in "Values" column

    :return model: fitted linear regression model
    """
    
    # Take only values series
    vals = dataframe['Values']
    
    # Convert to lagged form (as in the picture above)
    lagged_dataframe = pd.concat([vals.shift(2), vals.shift(1), vals], axis=1)
    
    # Rename columns
    lagged_dataframe.columns = ['Predictor 1', 'Predictor 2', 'Target']
    
    # Drop na rows (first two rows)
    lagged_dataframe.dropna(inplace=True)
    
    # Display first five rows
    print(lagged_dataframe.head(4))
    
    # Get linear model to train
    model = LinearRegression()
    
    # Prepare features and target for model
    x_train = lagged_dataframe[['Predictor 1', 'Predictor 2']]
    y_train = lagged_dataframe['Target']
    
    # Fit the model
    model.fit(x_train, y_train)
    return model


# Prepare simple time series
example_dataframe = pd.DataFrame({'Values':[5,7,9,7,5,5,3,4,6,14,6,3,5]})

# Get fitted model
fitted_model = simple_autoregression_function(example_dataframe)

# Now we can forecast the values by knowing 
# the current and previous values (2 predictors in total)
predictors = [3,5]
print(f'\nPredictors: {predictors}')
forecast = fitted_model.predict([predictors])

print(f'Forecasted value: {forecast}')