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
from pyspark.ml.feature import SQLTransformer
# $example off$
from pyspark.sql import SparkSession

def create_data():
    df = spark.createDataFrame([
        (0, 1.0, 3.0),
        (2, 2.0, 5.0)
    ], ["id", "v1", "v2"])
    return df

def pre_processing(df):
    ''' create tranform object, apply to df to generate another df '''
    sqlTrans = SQLTransformer(
        statement="SELECT *, (v1 + v2) AS v3, (v1 * v2) AS v4 FROM __THIS__")
    sqlTrans.transform(df).show()


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("SQLTransformerExample")\
        .getOrCreate()

    # create data
    df = create_data()
        
    # sql transform
    pre_processing(df)

    spark.stop()
