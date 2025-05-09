import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# Fonction qui supprime les outliers en utilisant la méthode IQR
def remove_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # On filtre les données qui sont à l'intérieur des bornes définies
    df_no_outliers = df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]
    
    return df_no_outliers

# Fonction qui applique le kmeans et qui nous permet de visualiser le clustering
def apply_kmeans(X, k_values=[2, 3], model_name="Dataset"):
    # Scale les donnees
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # On normalise les données
    
    # Pour chaque valeur de k
    for k in k_values:
        # On crée le modèle KMeans
        model = KMeans(n_clusters=k, random_state=0, n_init=10)
        model.fit(X_scaled)
        
        # On crée les prédictions
        predictions = model.predict(X_scaled)
        print(f"Inertie pour K={k} for {model_name}: {model.inertia_:.2f}")

        # Calculer l'Adjusted Rand Index si les vraies étiquettes existent (si disponibles)
        if "target" in dir(X):
            ari = adjusted_rand_score(X.target, predictions)
            print(f"Adjusted Rand Index for K={k} for {model_name}: {ari:.2f}")
        
        # Visualiser les données et les centroïdes
        plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=predictions, cmap='viridis', label='Data Points')
        plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids')
        plt.title(f'Clusters and Centroids for K={k} ({model_name})')
        plt.legend()
        plt.show()

# Loading iris dataset and making it a data frame
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# removing outliers (low, upper bound) from iris df
iris_df_no_outliers = remove_outliers(iris_df)
apply_kmeans(iris_df, model_name="Iris", k_values=[1,2,3,4,5,6,7,8,9,10])

# Breast cancer dataset and creating its data frame
cancer_data = load_breast_cancer()
cancer_df = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
cancer_df['target'] = cancer_data.target

# removing outliers (low, upper bound) from brest cancer
cancer_df_no_outliers = remove_outliers(cancer_df)
# apply_kmeans(cancer_df_no_outliers, model_name="Breast Cancer", k_values=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

# Wine dataset readnig from csv
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
wine_df = pd.read_csv(url, header=None)

# removing outliers (low, upper bound) from wine df
wine_df_no_outliers = remove_outliers(wine_df)
# apply_kmeans(wine_df_no_outliers, model_name="Wine", k_values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

# Loading mall customers .csv (located locally)
mall_customers_df = pd.read_csv("Mall_Customers.csv")

# Cleaning columns names, no more column snamed like: A B and more like: A_B
mall_customers_df.columns = mall_customers_df.columns.str.strip().str.replace(' ', '_')

# Add artificially values that are not there
mall_customers_df.loc[5:10, "Age"] = np.nan  # delete some ages
mall_customers_df.loc[15:20, "Annual_Income_(k$)"] = np.nan  # delete some revenues

# Imputation of the missing values by using moyenne
imputer = SimpleImputer(strategy="mean")
mall_customers_df[["Age", "Annual_Income_(k$)"]] = imputer.fit_transform(mall_customers_df[["Age", "Annual_Income_(k$)"]])

# We use numerical columns for the clustering
mall_customers_X = mall_customers_df[["Age", "Annual_Income_(k$)", "Spending_Score_(1-100)"]]

# We delete outliers from the mall customers
mall_customers_X_no_outliers = remove_outliers(mall_customers_X)

# Apply kmeans to mall customers
# apply_kmeans(mall_customers_X_no_outliers, model_name="Mall Customers", k_values=[2, 3, 4, 5, 6, 7])