
import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql import functions as f
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression


# Création du dataframe et extraction des données
spark : SparkSession = SparkSession.builder.getOrCreate()
data : DataFrame = spark.read.csv("gold1.csv", header = True)

# Nettoyage des données
data = data.na.drop()        # Retirer les colonnes vides
data = data.dropDuplicates() # Retirer les doublons


# Convertis les colonnes en int
data = data.withColumn("Date2",f.to_date(data["Date"]))                       # Ajoute une colonne "Date2" qui contient la date formattée
data = data.withColumn("Date2",f.unix_timestamp(data["Date2"])/86400)   # Modifie la colonne "Date2" pour la convertir en nombre de jours
data = data.withColumn("Date2",data.Date2.cast("int"))                  # Troncque le nombre de jours
data = data.withColumn("High",data.High.cast("int"))                    # Troncque le prix maximal par jour


# Préparation des données pour le modèle
assembler = VectorAssembler(inputCols=["Date2"],outputCol="features")
data_ready = assembler.transform(data).select("features","High").withColumnRenamed("High","label")

train_data, test_data = data_ready.randomSplit([0.8,0.2],seed=5)    # Séparation des données en entraînement et test à 80%/20%

# Régression linéaire
lr = LinearRegression(featuresCol="features",labelCol="label")
lr_model = lr.fit(train_data)



# Prédictions
test_predictions = lr_model.transform(test_data)
test_predictions.select("features","label","prediction").show()

# Évaluation
test_evaluation = lr_model.evaluate(test_data)
print("==================== Résultats ====================")
print("Erreur quadratique moyenne (MSE)",test_evaluation.meanSquaredError)
print("Erreur absolue moyenne (MAE)", test_evaluation.meanAbsoluteError)
print("Coefficient de Détermination (R2)", test_evaluation.r2)
print("===================================================")

# Transformer les données en format panda
train_panda = train_data.select("features","label").toPandas()
test_panda = test_data.select("features","label").toPandas()

# Création du graphique
plt.figure(figsize=(10,6))

train_panda["Date"] = train_panda["features"].apply(lambda x: x[0])
test_panda["Date"] = test_panda["features"].apply(lambda x: x[0])

plt.scatter(train_panda["Date"],train_panda["label"], color='blue', label = "Données d'entrainement")

plt.scatter(test_panda["Date"],test_panda["label"], color="green", label = "Données de test")

x_line = np.linspace(data.select("Date2").rdd.min()[0],data.select("Date2").rdd.max()[0],100)
y_line = lr_model.coefficients[0]* x_line + lr_model.intercept

plt.plot(x_line,y_line, color = "red",label= "Droite de régression")

plt.show()