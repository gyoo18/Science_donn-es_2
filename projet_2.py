
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

"""
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

data = spark.read.csv("gold1.csv")

data = data.na.drop()

data = data.dropDuplicates()
"""

df = pd.read_csv("gold1.csv")

df["Date2"] = pd.to_datetime(df["Date"])

df["Date2"] = df["Date2"].astype(int)
df["Date2"] = df["Date2"].astype(float)

df.dropna(axis=0, inplace=True)


print(df)
print(df.info())

df_donnees_numeriques = df.select_dtypes(include=["Float64"])
matrice_de_correlations = df_donnees_numeriques.corr()
sns.heatmap(matrice_de_correlations, annot=True, cmap='coolwarm')
plt.title('Corrélations entre les variables')
plt.show()




plt.scatter(df['Date2'], df['High'])
plt.show()

  