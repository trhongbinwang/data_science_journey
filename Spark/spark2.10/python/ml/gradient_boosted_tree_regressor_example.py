#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Gradient Boosted Tree Regressor Example.
"""
from __future__ import print_function

########## enviroment setup ################
import os
import sys

# set enviroment and path to run pyspark
spark_home = os.environ.get('SPARK_HOME', None)
print(spark_home)
if not spark_home:
    raise ValueError('SPARK_HOME environment variable is not set')
sys.path.insert(0, os.path.join(spark_home, 'python'))
sys.path.insert(0, os.path.join(spark_home, 'python/lib/py4j-0.10.4-src.zip')) ## may need to adjust on your system depending on which Spark version you're using and where you installed it.
##############################

# $example on$
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
# $example off$
from pyspark.sql import SparkSession


def load_data():
    # Load and parse the data file, converting it to a DataFrame.
    data = spark.read.format("libsvm").load("../../data/mllib/sample_libsvm_data.txt")
    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])
    return (data, trainingData, testData)
    
def make_pipeline(data):
    # Automatically identify categorical features, and index them.
    # Set maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer =\
        VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

    # Train a GBT model.
    gbt = GBTRegressor(featuresCol="indexedFeatures", maxIter=10)

    # Chain indexer and GBT in a Pipeline
    pipeline = Pipeline(stages=[featureIndexer, gbt])
    return pipeline
    
def predict_display(model, testData):
    # Make predictions.
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select("prediction", "label", "features").show(5)

    # Select (prediction, true label) and compute test error
    evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

    gbtModel = model.stages[1]
    print(gbtModel)  # summary only



if __name__ == "__main__":
    # initialize spark session
    spark = SparkSession\
        .builder\
        .appName("GradientBoostedTreeRegressorExample")\
        .getOrCreate()

    # load data
    (data, trainingData, testData) = load_data()
    # make pipeline
    pipeline = make_pipeline(data)
    
    # Train model.  This also runs the indexer.
    model = pipeline.fit(trainingData)
    # predict and display
    predict_display(model, testData)
    
    # stop
    spark.stop()
    
    
    
    
