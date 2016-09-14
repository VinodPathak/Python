# Databricks notebook source exported at Wed, 14 Sep 2016 13:03:52 UTC
irisdata = sqlContext.read.format("com.databricks.spark.csv").options(header='true',inferSchema='true').load('/FileStore/tables/jwfl958p1473832367404/iris.csv')

# COMMAND ----------

display(irisdata)

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression, GBTClassifier, NaiveBayes, RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import RegressionMetrics, BinaryClassificationMetrics
from pyspark.mllib.util import MLUtils

# COMMAND ----------

print irisdata.dtypes

# COMMAND ----------

stringIndexer = StringIndexer(inputCol="class", outputCol="label")
si_model = stringIndexer.fit(irisdata)
td = si_model.transform(irisdata)
new_data = td.drop('class')

# COMMAND ----------

display(new_data)

# COMMAND ----------

assembler = VectorAssembler(inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"], outputCol="features")

# COMMAND ----------

rfmodel = RandomForestClassifier()\
  .setLabelCol("label")\
  .setFeaturesCol("features")
#print (rfmodel.explainParams())

# COMMAND ----------

paramGrid = ParamGridBuilder().addGrid(rfmodel.maxBins, [10,20]).addGrid(rfmodel.maxDepth, [5,10]).build()
pipeline = Pipeline().setStages([assembler,rfmodel])
evaluator = MulticlassClassificationEvaluator()
tvs = CrossValidator().setEstimator(pipeline).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(4)

# COMMAND ----------

training, test = new_data.randomSplit([0.75, 0.25], seed = 12345)
model = tvs.fit(training)

# COMMAND ----------

from pyspark.sql import Row
newtest = Row(sepal_length=3.50, sepal_width=1.0, petal_length=2.00, petal_width=0.30)
df4 = sc.parallelize([newtest]).toDF()
dff = model.transform(df4)
display(dff)

# COMMAND ----------

from pyspark.sql.types import *
schema = StructType([StructField('sepal_length', FloatType(), True),
                     StructField('sepal_width', FloatType(), True),
                     StructField('petal_length', FloatType(), True),
                    StructField('petal_width', FloatType(), True)])
