import pandas as pd
import csv
import matplotlib.pyplot as plt


def fill_missing_data(city_data):
    
    with open(f'DATA\weather_training_{city_data}.csv') as csv_file:
        city = pd.read_csv(csv_file)
    
    with open(f'DATA\weather_test_{city_data}.csv') as csv_file:
        city_test = pd.read_csv(csv_file)
    wind = ["9am wind speed (km/h)", "3pm wind speed (km/h)"]
    for feature in wind:
        for index in city.index:
            if city.loc[index, feature] == 'Calm':
                city.loc[index, feature] = float(0)
            elif not pd.isnull(city.loc[index, feature]):
                city.loc[index, feature] = float(city.loc[index, feature])
        for index in city_test.index:
            if city_test.loc[index, feature] == 'Calm':
                city_test.loc[index, feature] = float(0)
            elif not pd.isnull(city_test.loc[index, feature]):
                city_test.loc[index, feature] = float(city_test.loc[index, feature])
    
    summer_months = ['12', '01', '02']
    autumn_months = ["03", "04", "05"]
    winter_months = ["06", "07", "08"]
    spring_months = ["09", "10", "11"]
    
    summer_ind = []
    autumn_ind = []
    winter_ind = []
    spring_ind = []
    test_summer_ind = []
    test_autumn_ind = []
    test_winter_ind = []
    test_spring_ind = []

    for i in city.index:
        if city.loc[i, 'Date'][2:4] in summer_months:
            summer_ind.append(i)
        elif city.loc[i, 'Date'][2:4] in autumn_months:
            autumn_ind.append(i)
        elif city.loc[i, 'Date'][2:4] in winter_months:
            winter_ind.append(i)
        elif city.loc[i, 'Date'][2:4] in spring_months:
            spring_ind.append(i)
    for i in city_test.index:
        if city_test.loc[i, 'Date'][2:4] in summer_months:
            test_summer_ind.append(i)
        elif city_test.loc[i, 'Date'][2:4] in autumn_months:
            test_autumn_ind.append(i)
        elif city_test.loc[i, 'Date'][2:4] in winter_months:
            test_winter_ind.append(i)
        elif city_test.loc[i, 'Date'][2:4] in spring_months:
            test_spring_ind.append(i)
    
    features = ["Minimum temperature (C)", "Maximum temperature (C)", "9am Temperature (C)", "3pm Temperature (C)", 
                "Sunshine (hours)", "Evaporation (mm)", "9am wind speed (km/h)", "3pm wind speed (km/h)", 
                "Speed of maximum wind gust (km/h)", "3pm MSL pressure (hPa)", "9am MSL pressure (hPa)"]
    if city_data == 'adelaide':
        features.remove('Sunshine (hours)')
        features.remove('Evaporation (mm)')
    elif city_data == 'brisbane':
        features.remove('Evaporation (mm)')

    for feature in features:
        summer_avg = city.loc[summer_ind, feature].mean()
        autumn_avg = city.loc[autumn_ind, feature].mean()
        winter_avg = city.loc[winter_ind, feature].mean()
        spring_avg = city.loc[spring_ind, feature].mean()
        for index in city.index:
            if pd.isnull(city.loc[index, feature]):
                if index in summer_ind:
                    city.loc[index, feature] = summer_avg
                elif index in autumn_ind:
                    city.loc[index, feature] = autumn_avg
                elif index in spring_ind:
                    city.loc[index, feature] = winter_avg
                else:
                    city.loc[index, feature] = spring_avg
        for index in city_test.index:
            if pd.isnull(city_test.loc[index, feature]):
                if index in test_summer_ind:
                    city_test.loc[index, feature] = summer_avg
                elif index in test_autumn_ind:
                    city_test.loc[index, feature] = autumn_avg
                elif index in test_spring_ind:
                    city_test.loc[index, feature] = winter_avg
                else:
                    city_test.loc[index, feature] = spring_avg
    
    # city.to_csv(f"weather_{city_data}_filled.csv")

    non_season_data = ["Rainfall (mm)", "9am relative humidity (%)", "3pm relative humidity (%)",
                        "9am cloud amount (oktas)", "3pm cloud amount (oktas)"]
    if city_data == 'adelaide':
        non_season_data.remove("9am cloud amount (oktas)")
        non_season_data.remove("3pm cloud amount (oktas)")

    for feature in non_season_data:
        city[feature].fillna(city[feature].mean(), inplace=True)
        city_test[feature].fillna(city[feature].mean(), inplace=True)





    all_features = non_season_data + features
    numeric_features = [column for column in city.columns if column != 'Date']
    if city_data == 'melbourne':
        feature_left = ['9am Temperature (C)', 'Evaporation (mm)', '3pm MSL pressure (hPa)', '9am relative humidity (%)', '3pm wind speed (km/h)', 'Sunshine (hours)']
    elif city_data == 'adelaide':
        feature_left = ['9am Temperature (C)', '9am relative humidity (%)', '3pm MSL pressure (hPa)', 'Rainfall (mm)']
    elif city_data == 'brisbane':
        feature_left = ['Minimum temperature (C)', '3pm MSL pressure (hPa)', '9am cloud amount (oktas)', 'Rainfall (mm)', 'Speed of maximum wind gust (km/h)']
    elif city_data == 'sydney' :
        feature_left = ['Minimum temperature (C)', 'Evaporation (mm)','3pm relative humidity (%)', '3pm wind speed (km/h)', '9am relative humidity (%)', '9am cloud amount (oktas)']
    feature_left.append('avg_demand')
    for feature in numeric_features:
        if feature not in feature_left:
            city.drop(feature, inplace=True, axis=1)
            city_test.drop(feature, inplace=True, axis=1)
    city.to_csv(f"DATA\weather_filled_training_{city_data}.csv")
    city_test.to_csv(f"DATA\weather_filled_test_{city_data}.csv")
    return


fill_missing_data("melbourne")
fill_missing_data("brisbane")
fill_missing_data("sydney")
