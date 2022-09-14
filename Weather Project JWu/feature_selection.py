import pandas as pd
import os
import csv
import random
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import normalized_mutual_info_score
import numpy as np

"""
Feature filtering based on MI for regression.
"""

def select_features(city_data):
    
    filtered_features = []
    
    with open(f"DATA\weather_filled_training_{city_data}.csv") as csv_file:
        city = pd.read_csv(csv_file)
    city.drop("Unnamed: 0", inplace = True, axis = 1)
    numeric_features = [column for column in city.columns if (column != "Date" and column != "avg_demand")]
    equal_freq_6 = KBinsDiscretizer(n_bins= 6, encode = 'ordinal', strategy = 'quantile')
    mi_score = []
    for feature in numeric_features:
        # compute MI between feature and average avg_demand
        feature_one = city[feature]
        feature_two = city['avg_demand']
        feature_one = pd.DataFrame(feature_one)
        feature_two = pd.DataFrame(feature_two)
        feature_one[f'binned {feature}'] = equal_freq_6.fit_transform(city[[feature]]).astype(int)
        feature_two[f'binned avg_demand'] = equal_freq_6.fit_transform(city[['avg_demand']]).astype(int)
        score = normalized_mutual_info_score(feature_one[f'binned {feature}'], feature_two[f'binned avg_demand'], average_method='min')
        mi_score.append([feature, 'avg_demand', score])
    print("================================================================================================================================================================")
    print("================================================================================================================================================================")
    print("================================================================================================================================================================")
    print(city_data)
    mi_score = sorted(mi_score, key=lambda x: (x[2]), reverse=True)
    for i in range(len(mi_score)):
        print(mi_score[i])
    print("================================================================================================================================================================")
    print("================================================================================================================================================================")
    print("================================================================================================================================================================")
    return 
select_features('melbourne')
select_features('brisbane')
select_features('sydney')





