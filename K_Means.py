import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, -2:].values

# #Within-Cluster Sum of Square
# wcss = []

# for i in range(1,11):
#     kmeans = KMeans(n_clusters = i,init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
#     kmeans.fit(X)
#     wcss.append(kmeans.inertia_)
    
# #Elbow Method Plot    
# plt.plot(range(1,11), wcss)
# plt.title('The Elbow Method')
# plt.xlabel('Number of Clusters')
# plt.ylabel('WCSS')
# plt.show()

#Initialize K-Means to Mall_Customers Dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

#Visual
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0,1], s=100, c='red', label='Frugal')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1,1], s=100, c='blue', label='Medium')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2,1], s=100, c='green', label='High')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3,1], s=100, c='cyan', label='OverSpending')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4,1], s=100, c='magenta', label='Low')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='yellow', label='Centroids')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()