import pandas as pd
import csv
import matplotlib.pyplot as plt




def plots(city_data):
    
    with open(f'DATA\weather_{city_data}.csv') as csv_file:
        city = pd.read_csv(csv_file)

    
    feature_list = ["Rainfall (mm)", "Sunshine (hours)", "Evaporation (mm)", "9am cloud amount (oktas)"]

    am_features = ["9am relative humidity (%)", "9am Temperature (°C)", "9am MSL pressure (hPa)"]

    pm_features = ["3pm relative humidity (%)", "3pm Temperature (°C)", "3pm MSL pressure (hPa)"]

    plt.close()
    plt.scatter(range(len(city["Maximum temperature (°C)"])), city["Maximum temperature (°C)"])
    plt.scatter(range(len(city["Minimum temperature (°C)"])), city["Minimum temperature (°C)"])
    plt.title(f'{city_data}: Max and Min Temperatures')
    plt.xlabel("Days after 1/02/2021")
    plt.ylabel("Temperature (°C)")
    plt.legend(["Maximum temperature (°C)", "Minimum temperature (°C)"])
    plt.savefig(f"Max and Min Temperatures {city_data}.png")
    plt.close()
    

    for feature in feature_list:
        plt.close()
        plt.scatter(range(len(city[feature])), city[feature])
        plt.title(f'{city_data}: {feature}')
        plt.xlabel("Days after 1/02/2021")
        plt.ylabel(feature)
        plt.savefig(f"{feature} {city_data}.png")
        plt.close()

    for i in range(len(am_features)):
        plt.close()
        f = am_features[i]
        name = f[4:]
        am = am_features[i]
        pm = pm_features[i]
        plt.scatter(range(len(city[am])), city[am])
        plt.scatter(range(len(city[pm])), city[pm])
        plt.legend([am, pm])
        plt.title(f'{city_data}: 9am and 3pm {name}')
        plt.xlabel("Days after 1/02/2021")
        plt.ylabel(name)
        plt.savefig(f"{name} (9am & 3pm) {city_data}.png")
        plt.close()
    

    return

plots("Melbourne")
plots("Sydney")
plots("Adelaide")
plots("Brisbane")
