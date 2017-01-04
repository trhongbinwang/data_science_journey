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
from pyspark.ml.clustering import BisectingKMeans
# $example off$
from pyspark.sql import SparkSession

"""
An example demonstrating bisecting k-means clustering.
Run with:
  bin/spark-submit examples/src/main/python/ml/bisecting_k_means_example.py
"""

def load_data():
    # Loads data.
    dataset = spark.read.format("libsvm").load("../../data/mllib/sample_kmeans_data.txt")
    return dataset

def train_model(dataset):
    # Trains a bisecting k-means model.
    bkm = BisectingKMeans().setK(2).setSeed(1)
    model = bkm.fit(dataset)
    return model

def evaluate(model):
    # Evaluate clustering.
    cost = model.computeCost(dataset)
    print("Within Set Sum of Squared Errors = " + str(cost))

def display(model):
    # Shows the result.
    print("Cluster Centers: ")
    centers = model.clusterCenters()
    for center in centers:
        print(center)
    

if __name__ == "__main__":
    # initialize spark session
    spark = SparkSession\
        .builder\
        .appName("BisectingKMeansExample")\
        .getOrCreate()

    # load data
    dataset = load_data()
    # training model
    model = train_model(dataset)
    # evaluate
    evaluate(model)
    # display
    display(model)

    # stop
    spark.stop()
