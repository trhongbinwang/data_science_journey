# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 15:56:03 2016
spark python summary


"""
# create spark enviroment
    sc = SparkContext(appName="name")
    sql_sc = SQLContext(sc)
    # read in data
    p_df = pd.read_csv('current.csv')
    # create spark df from pandas df
    df = sql_sc.createDataFrame(p_df_str)
    # change the column name 
    df = df.withColumnRenamed("st1", "st")
    # take action
    print(df.count()) # 
    # schema
    df.printSchema()
    # filter in spark df
    df = df.filter(df['f'] == 1) \
            .select('name', 'source', 'time')
    # convert spark df to pandas df
    pdf = df.toPandas()

