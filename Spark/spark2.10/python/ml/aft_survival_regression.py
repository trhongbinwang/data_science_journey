'''
focus on ml folder not mllib

'''
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
from pyspark.ml.regression import AFTSurvivalRegression
from pyspark.ml.linalg import Vectors
# $example off$
from pyspark.sql import SparkSession

"""
An example demonstrating aft survival regression.
Run with:
  bin/spark-submit examples/src/main/python/ml/aft_survival_regression.py
"""

def create_data(spark):
    # create df from a list of tuple, scheme = a list of column names
    # Creates a DataFrame from an RDD, a list or a pandas.DataFrame (only 3 ways)
    training = spark.createDataFrame([
        (1.218, 1.0, Vectors.dense(1.560, -0.605)),
        (2.949, 0.0, Vectors.dense(0.346, 2.158)),
        (3.627, 0.0, Vectors.dense(1.380, 0.231)),
        (0.273, 1.0, Vectors.dense(0.520, 1.151)),
        (4.199, 0.0, Vectors.dense(0.795, -0.226))], ["label", "censor", "features"])
    return training

def train_model(training): 
    quantileProbabilities = [0.3, 0.6]
    aft = AFTSurvivalRegression(quantileProbabilities=quantileProbabilities,
                                quantilesCol="quantiles")
    model = aft.fit(training)
    return model    

def display(model, training):
    # Print the coefficients, intercept and scale parameter for AFT survival regression
    print("Coefficients: " + str(model.coefficients))
    print("Intercept: " + str(model.intercept))
    print("Scale: " + str(model.scale))
    model.transform(training).show(truncate=False)
    

if __name__ == "__main__":
    # initialize the spark session
    spark = SparkSession \
        .builder \
        .appName("AFTSurvivalRegressionExample") \
        .getOrCreate()
    # call underly sparkContext to config
    spark.sparkContext.setLogLevel('ERROR')

    # load data
    training = create_data(spark)
    # define the model and train
    model = train_model(training)
    # display
    display(model, training)
    
    # stop spark session
    spark.stop()
