import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

r2_list = []
mse_list = []
def pca_function(city_data):
    with open (f"DATA/s_filled_training_{city_data}.csv") as csv_file:
        city_train = pd.read_csv(csv_file)
    with open (f"DATA/s_filled_test_{city_data}.csv") as csv_file:
        city_test = pd.read_csv(csv_file)

    city_train.drop(columns = ["Unnamed: 0" ,"Date"], inplace = True)
    city_test.drop(columns = ["Unnamed: 0" ,"Date"], inplace = True)

    # From the explained_variance is can be seen that PCA with 2 dimensions capture, on average, 99.7%+ of the variance
    pca = PCA(n_components=4)

    X_train = city_train.drop(columns = ["avg_demand"])
    Y_train = city_train["avg_demand"]
    X_test = city_test.drop(columns = ["avg_demand"])
    Y_test = city_test["avg_demand"]

    # Scaling our features
    normalise_scaler = MinMaxScaler()

    X_train = normalise_scaler.fit_transform(X_train)
    X_test = normalise_scaler.fit_transform(X_test)

    X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=0.2, random_state=0)

    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    X_validation = pca.transform(X_validation)

    regressor = LinearRegression()
    regressor.fit(X_train, Y_train)
    Y_prediction = regressor.predict(X_test)
    val_prediction = regressor.predict(X_validation)


    explained_variance = pca.explained_variance_ratio_

    print("Variance Explained: ", explained_variance)
    print("Total variance: ", sum(explained_variance))

    r2 = regressor.score(X_test, Y_test)
    mse = mean_squared_error(Y_test, Y_prediction)
    r2_list.append(r2)
    mse_list.append(mse)

    print(f'{city_data} test')
    print('R2', r2)
    print('MSE', mse, '\n')

    r2 = regressor.score(X_validation, Y_validation)
    mse = mean_squared_error(Y_validation, val_prediction)

    print(f'{city_data} validation')
    print('R2', r2)
    print('MSE', mse)   

    print("\n============================================\n")

    # Perform 2-means clustering
    clusters = KMeans(n_clusters=2).fit(X_train)

    # Visualise the first 2 PCs
    sns.scatterplot(x=X_train[:,0], 
                    y=X_train[:,1],
                    hue=clusters.labels_)
    plt.title("PCA with 2 components")
    plt.xlabel('1st Principal Component')
    plt.ylabel('2nd Principal Component')

    plt.savefig(f"pca{city_data}.jpg")
    plt.close()
    
    return

pca_function("melbourne")
pca_function("brisbane")
pca_function("sydney")


