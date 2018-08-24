import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt




pd.set_option('display.max_columns', 1000)


df = pd.read_csv('College_Data', index_col=0)

print(df.head())
print(df.info())
print(df.describe())


sns.lmplot(x='Room.Board', y='Grad.Rate', data=df, hue='Private', palette='coolwarm', size=6)
plt.show()

# Scatter plot
sns.lmplot(x='Outstate', y='F.Undergrad', data=df, hue='Private', size=6)
plt.show()

# Stacked Histogram
g = sns.FacetGrid(df,hue='Private',palette='coolwarm')
g = g.map(plt.hist, 'Outstate', bins=20, alpha=.7)
plt.show()

g2 = sns.FacetGrid(df,hue='Private',palette='coolwarm')
g2 = g2.map(plt.hist, 'Grad.Rate', bins=20, alpha=.7)
plt.show()



print(df[df['Grad.Rate']>100])

df['Grad.Rate']['Cazenovia College'] = 100

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2)

kmeans.fit(df.drop('Private', axis=1))

print(kmeans.cluster_centers_)

def converter(private):
    if private == 'Yes':
        return 1
    else:
        return 0

df['Cluster'] = df['Private'].apply(converter)

print(df.head())


from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(df['Cluster'], kmeans.labels_))
print('\n')
print(classification_report(df['Cluster'], kmeans.labels_))