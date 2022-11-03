import pandas as pd
import os
import csv
import random

def split_test_set(city_data):
    with open(f'DATA\weather_{city_data}.csv') as csv_file:
        city = pd.read_csv(csv_file)
    city.drop("Unnamed: 0", inplace = True, axis = 1) 
    city = city.drop(range(410, 424))
    with open(f'DATA\price {city_data}.csv') as csv_file:
        demand = pd.read_csv(csv_file)
    city['avg_demand'] = demand['avg_demand']
    test_set_ind = []
    while len(test_set_ind) < 82:
        index = random.randint(0, 409)
        while (True):
            if (index in test_set_ind):
                index = random.randint(0, 409)
            else:
                test_set_ind.append(index)
                break
    test_set_ind = sorted(test_set_ind)
    training_set_ind = []
    for i in range(410):
        if i not in test_set_ind:
            training_set_ind.append(i)
    test_set_df = city.iloc[test_set_ind[0]:test_set_ind[0]+1]
    test_set_df = test_set_df.reset_index(drop=True)
    for i in range(1, len(test_set_ind)):
        index = test_set_ind[i]
        curr_df = city.iloc[index:index+1]
        test_set_df = pd.concat([test_set_df, curr_df], ignore_index=True)

    training_set_df = city.iloc[training_set_ind[0]:training_set_ind[0]+1]
    training_set_df = training_set_df.reset_index(drop=True)
    for i in range(1, len(training_set_ind)):
        index = training_set_ind[i]
        curr_df = city.iloc[index:index+1]
        training_set_df = pd.concat([training_set_df, curr_df], ignore_index=True)
    
    delete_cols = ['3pm wind direction','Direction of maximum wind gust ', 'Time of maximum wind gust', 
                '9am wind direction']
    training_set_df.drop(columns=delete_cols, inplace=True, axis=1)
    test_set_df.drop(columns=delete_cols, inplace=True, axis=1)
    test_set_df = test_set_df.reindex()
    training_set_df = training_set_df.reindex()

    if city_data == 'brisbane':
        training_set_df.drop(columns=['Evaporation (mm)'], inplace=True, axis=1)
        test_set_df.drop(columns=['Evaporation (mm)'], inplace=True, axis=1)
    training_set_df.to_csv(f'DATA\weather_training_{city_data}.csv')
    test_set_df.to_csv(f'DATA\weather_test_{city_data}.csv')
    return
split_test_set('melbourne')
split_test_set('sydney')
split_test_set('brisbane')
