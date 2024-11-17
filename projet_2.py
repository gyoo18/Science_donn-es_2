
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression


#Creation dataframe
spark = SparkSession.builder.getOrCreate()
data = spark.read.csv("gold1.csv", header = True)

#Clear dataframe
data = data.na.drop()
data = data.dropDuplicates()


# cast column into int
data = data.withColumn("Date2",f.to_date("Date"))
data = data.withColumn("Date2",f.unix_timestamp(data["Date2"])/86400)
data = data.withColumn("Date2",data.Date2.cast("int"))
data = data.withColumn("High",data.High.cast("int"))


#Preparation donnees pour modele
assembler = VectorAssembler(inputCols=["Date2"],outputCol="features")
data_ready = assembler.transform(data).select("features","High").withColumnRenamed("High","label")

train_data, test_data = data_ready.randomSplit([0.8,0.2],seed=5)

#regression lineaire
lr = LinearRegression(featuresCol="features",labelCol="label")
lr_model = lr.fit(train_data)

#Predictions
test_predictions = lr_model.transform(test_data)
test_predictions.select("features","label","prediction").show()

# Evaluation
test_evaluation = lr_model.evaluate(test_data)
print("root mean square Error",test_evaluation.meanSquaredError)
print("mean absolute error", test_evaluation.meanAbsoluteError)





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

sns.lmplot(x="Date2",y="High",data = df)

plt.show()
"""