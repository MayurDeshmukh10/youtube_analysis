import pandas as pd
import numpy as np
import json
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import plotly.offline as py

import plotly.graph_objs as go
from plotly.graph_objs import *
import seaborn as sns


df = pd.read_csv('INvideos.csv')
df.head()


df.dtypes


cluster = df[['likes','dislikes', 'views', 'comment_count']]


from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(cluster)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

df2 = cluster.apply(lambda x:(x.astype(float) - min(x))/(max(x)-min(x)), axis = 0)
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(df2)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


kmeans = KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans = kmeans.fit_predict(df2)
df2['cluster']=y_kmeans

trace1 = go.Scatter3d(
    x = df2['likes'].values,
    y = df2['comment_count'].values,
    z = df2['views'].values,
    mode='markers',
    marker=dict(
        size=12,
        color=df2['cluster'].values,# set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )

)

data = [trace1]
layout = go.Layout(
    scene=Scene(
        xaxis=XAxis(title='Likes'),
        yaxis=YAxis(title='Comment'),
        zaxis=ZAxis(title='Views')
        ),
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    ),
)

fig = go.Figure(data=data, layout=layout)
py.plot(fig, filename='3d-scatter-colorscale')
