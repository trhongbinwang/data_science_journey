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


from pyspark.sql import SparkSession
# $example on$
from pyspark.ml.feature import Bucketizer
# $example off$

def create_data():
    data = [(-999.9,), (-0.5,), (-0.3,), (0.0,), (0.2,), (999.9,)]
    dataFrame = spark.createDataFrame(data, ["features"])
    return dataFrame

def pre_processing(dataFrame):
    
    splits = [-float("inf"), -0.5, 0.0, 0.5, float("inf")]

    bucketizer = Bucketizer(splits=splits, inputCol="features", outputCol="bucketedFeatures")

    # Transform original data into its bucket index.
    bucketedData = bucketizer.transform(dataFrame)

    print("Bucketizer output with %d buckets" % (len(bucketizer.getSplits())-1))
    bucketedData.show()
    
    

if __name__ == "__main__":
    # initialize
    spark = SparkSession\
        .builder\
        .appName("BucketizerExample")\
        .getOrCreate()
    
    # create data
    dataFrame = create_data()
    # pre_processing
    pre_processing(dataFrame)
    
    # stop
    spark.stop()
