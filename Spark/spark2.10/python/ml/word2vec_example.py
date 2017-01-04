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
from pyspark.ml.feature import Word2Vec
# $example off$
from pyspark.sql import SparkSession

#def create_data():
#    # Input data: Each row is a bag of words from a sentence or document.
#    documentDF = spark.createDataFrame([
#        ("Hi I heard about Spark".split(" "), ),
#        ("I wish Java could use case classes".split(" "), ),
#        ("Logistic regression models are neat".split(" "), )
#    ], ["text"])
#    return documentDF

def create_data():
    # Input data: Each row is a bag of words from a sentence or document.
    # one element tuple (1, ). trailing comma
    sent = ("Hi I heard about Spark".split(" "),)
    print(sent)
    print(type(sent)) # tuple
    documentDF = spark.createDataFrame([
        ("Hi I heard about Spark".split(" "),),
        ("I wish Java could use case classes".split(" "),),
        ("Logistic regression models are neat".split(" "),)
    ], ["text"])
    return documentDF

    
def pre_processing(documentDF):
    ''' word2vec '''
    # Learn a mapping from words to Vectors.
    word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="text", outputCol="result")
    model = word2Vec.fit(documentDF)

    result = model.transform(documentDF)
    for row in result.collect():
        text, vector = row
        print("Text: [%s] => \nVector: %s\n" % (", ".join(text), str(vector)))


if __name__ == "__main__":
    
    spark = SparkSession\
        .builder\
        .appName("Word2VecExample")\
        .getOrCreate()

    # create data
    documentDF = create_data()
    # word2vec
    pre_processing(documentDF)

    spark.stop()
