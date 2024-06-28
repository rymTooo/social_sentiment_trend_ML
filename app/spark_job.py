import subprocess
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import struct, col
import mlflow 
import os
import pandas

# might need to change localhost to something else
spark = SparkSession \
    .builder \
    .appName("StructuredNetworkWordCount") \
    .config("spark.driver.host", "spark-master")\
    .getOrCreate()

#**** may use read instead of readstream ****
# df = spark \
#   .readStream \
#   .format("kafka") \
#   .option("kafka.bootstrap.servers", "kafka:29092") \
#   .option("subscribe", "test-topic") \
#   .load()

# df = df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)")
# print("this is df columns : ", df.columns)
# print("this is df type >>>>>>>>>", type(df))

os.environ['AWS_ACCESS_KEY_ID'] = 'CWRUnvN2zh8rqE7pidsw'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'V8RfUWQlnB4QUa7rGHbvfHjhjLiOutRa8AZ9TPvy'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://172.20.0.2:9000' #'http://172.20.0.2:9000' # http://localhost:9000

# Set the tracking URI to your MLflow server
mlflow.set_tracking_uri("http://172.20.0.3:5000")  # Replace with your MLflow server URI

# Specify the model URI using model registry
model_name = "NB_model"
model_version = 3
model_uri = f"models:/{model_name}/{model_version}"  # Replace with your model name and version

#load dependencies
# dependencies = mlflow.pyfunc.get_model_dependencies(model_uri)
# print("type ", type(dependencies))
# print("this is dependencies ", dependencies)
# subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", dependencies])

# >>> load and prediction method for spark ML lib
# loaded_model = mlflow.spark.load_model(model_uri, spark=spark)
# predictions = loaded_model.predict(df)
# print(predictions)
test_df = pandas.DataFrame({"text":["this is a sample positive tweet", "and this is negative :("]})
test_df = spark.createDataFrame(test_df)
print("this is test df type >>>>>>>>>>", type(test_df))
print("this is test df columns >>>>>>>", test_df.columns)
loaded_model = mlflow.pyfunc.spark_udf(
    spark, 
    model_uri=model_uri,
    env_manager="local"
    )
print("-----------load model success----------------")
# Predict using the loaded model
predictions = test_df.withColumn('predictions', loaded_model(col('text')))
print("-----------prediction success----------------")
# predictions = df.withColumn('predictions', loaded_model(struct(*map(col, df.columns))))
predictions.printSchema()
print(predictions)


# result variable(maybe a dataframe) . write stream / write . output mode . format . start
# query = predictions \
#     .writeStream\
#     .outputMode("append") \
#     .format("console") \
#     .start()
#     # .option("kafka.bootstrap.servers", "kafka:29092") \
#     # .option("topic", "receive-topic") \
#     # .option("checkpointLocation", "/tmp/checkpoints") \
# query.awaitTermination()

predictions \
    .write \
    .format("console") \
    .option("truncate", False) \
    .save()

spark.stop()
