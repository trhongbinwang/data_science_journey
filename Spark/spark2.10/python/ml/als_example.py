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

# import sys
if sys.version >= '3':
    long = int

from pyspark.sql import SparkSession


from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row


def load_data(spark):
    '''' read df from RDD '''
    # windows change from \ to /
#    fileName = '/home/hongbin/git/data_science_journey/Spark/spark2.10/data/mllib/als/sample_movielens_ratings.txt'
    fileName = '../../data/mllib/als/sample_movielens_ratings.txt'
    lines = spark.read.text(fileName).rdd
    parts = lines.map(lambda row: row.value.split("::"))
    ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                         rating=float(p[2]), timestamp=long(p[3])))
    ratings = spark.createDataFrame(ratingsRDD)
    (training, test) = ratings.randomSplit([0.8, 0.2])
    return (training, test)

def training_model(training):
    # Build the recommendation model using ALS on the training data
    als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating")
    model = als.fit(training)
    return model
    
def evaluation(model, test):
    # Evaluate the model by computing the RMSE on the test data
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = " + str(rmse))
    

if __name__ == "__main__":
    # initialize the spark session
    spark = SparkSession\
        .builder\
        .appName("ALSExample")\
        .getOrCreate()
        
    spark.sparkContext.setLogLevel('ERROR')
    # load data
    (training, test) = load_data(spark)
    # training model
    model = training_model(training)
    # evaluation
    evaluation(model, test)
    # stop spark session
    spark.stop()
