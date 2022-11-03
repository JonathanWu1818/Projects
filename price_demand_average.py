import pandas as pd
import csv
import matplotlib.pyplot as plt

def average_price_demand():

    with open("DATA\price_demand_data.csv") as csv_file:
        price_demand = pd.read_csv(csv_file)

    rows = []

    totalprice = 0
    num_data = 0
    i = 0
    while i < len(price_demand) - 1:

        if price_demand.iloc[i]["SETTLEMENTDATE"][:-9] == price_demand.iloc[i+1]["SETTLEMENTDATE"][:-9]:
            totalprice = totalprice + price_demand.iloc[i]["TOTALDEMAND"]
            num_data = num_data + 1
            i = i + 1
        
        else:
            totalprice = totalprice + price_demand.iloc[i]["TOTALDEMAND"]
            num_data = num_data + 1
            i = i + 1
            totalprice = totalprice + price_demand.iloc[i]["TOTALDEMAND"]
            num_data = num_data + 1

            rows.append({
                "date": price_demand.iloc[i-1]["SETTLEMENTDATE"][:-9],
                "avg_demand": totalprice/num_data,
                "state": price_demand.iloc[i]["REGION"]
            })

            num_data = 0
            totalprice = 0
            i = i + 1
            

    df = pd.DataFrame(rows)

    sydney = df.loc[df["state"] == 'NSW1']
    melbourne = df.loc[df["state"] == 'VIC1']
    brisbane = df.loc[df["state"] == 'QLD1']
    adelaide = df.loc[df["state"] == 'SA1']
    
    print(sydney)
    sydney.to_csv("price sydney1")

    
    sydney.to_csv("DATA\price sydney.csv", index = False)
    melbourne.to_csv("DATA\price melbourne.csv", index = False)
    brisbane.to_csv("DATA\price brisbane.csv", index = False)



    return

average_price_demand()


