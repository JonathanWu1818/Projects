import csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import normalized_mutual_info_score
import pandas as pd
from itertools import permutations

def feature_scaling(city_data):
    
    with open (f"DATA\weather_filled_training_{city_data}.csv") as csv_file:
        city = pd.read_csv(csv_file)
    with open (f"DATA\weather_filled_test_{city_data}.csv") as csv_file:
        city_test = pd.read_csv(csv_file)
    city.drop("Unnamed: 0", inplace = True, axis = 1) 
    city_test.drop("Unnamed: 0", inplace = True, axis = 1) 
    numeric_features = [column for column in city.columns if column not in ['Date', 'avg_demand']]
    
    # print(numeric_features)
    
    normalise_scaler = MinMaxScaler()

    city[numeric_features] = normalise_scaler.fit_transform(city[numeric_features])
    city_test[numeric_features] = normalise_scaler.fit_transform(city_test[numeric_features])

    # after selection
    if city_data == 'melbourne':
        feature_left = ['9am Temperature (C)', 'Evaporation (mm)', '3pm MSL pressure (hPa)', '9am relative humidity (%)', '3pm wind speed (km/h)', 'Sunshine (hours)']
    elif city_data == 'brisbane':
        feature_left = ['Minimum temperature (C)', '3pm MSL pressure (hPa)', '9am cloud amount (oktas)', 'Rainfall (mm)', 'Speed of maximum wind gust (km/h)']
    elif city_data == 'sydney' :
        feature_left = ['Minimum temperature (C)', 'Evaporation (mm)','3pm relative humidity (%)', '3pm wind speed (km/h)', '9am relative humidity (%)', '9am cloud amount (oktas)']
    feature_left.append('avg_demand')
    for feature in numeric_features:
        if feature not in feature_left:
            city.drop(feature, inplace=True, axis=1)
            city_test.drop(feature, inplace=True, axis=1)
            
    city.to_csv(f"DATA\s_filled_training_{city_data}.csv")
    city_test.to_csv(f"DATA\s_filled_test_{city_data}.csv")
    return

feature_scaling("melbourne")
feature_scaling("sydney")
feature_scaling("brisbane")





