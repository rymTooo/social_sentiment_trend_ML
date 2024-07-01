import subprocess
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import struct, col
import mlflow 
import os
import pandas
import nltk

# might need to change localhost to something else
spark = SparkSession \
    .builder \
    .appName("StructuredNetworkWordCount") \
    .config("spark.driver.host", "spark-master")\
    .getOrCreate()


nltk.download('stopwords')
#**** may use read instead of readstream ****
df = spark \
  .readStream \
  .format("kafka") \
  .option("kafka.bootstrap.servers", "172.20.0.5:29092") \
  .option("subscribe", "test-topic") \
  .load()

print("this is input df type >> ", type(df))

df = df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)")
# print("this is df columns : ", df.columns)
# print("this is df type >>>>>>>>>", type(df))

os.environ['AWS_ACCESS_KEY_ID'] = 'CWRUnvN2zh8rqE7pidsw'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'V8RfUWQlnB4QUa7rGHbvfHjhjLiOutRa8AZ9TPvy'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://172.20.0.2:9000' #minio api ip:port

# Set the tracking URI to your MLflow server
mlflow.set_tracking_uri("http://172.20.0.3:5000")  # Replace with your MLflow server URI

# Specify the model URI using model registry
model_name = "NB_model"
model_version = 3
model_uri = f"models:/{model_name}/{model_version}"  # Replace with your model name and version
model_path = "./naiveBayes_model.py"
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
# test_df = pandas.read_csv("sample_data.csv", encoding='utf-8')
# test_df = spark.createDataFrame(test_df)
print("this is test df type >>>>>>>>>>", type(test_df))
# print("this is test df schema>>>>>>>")
# test_df.printSchema()

loaded_model = mlflow.pyfunc.spark_udf(
    spark, 
    model_uri=model_uri,
    env_manager="local"
    )
print("-----------load model success----------------")

# Predict using the loaded model
predictions = df.withColumn('predictions', loaded_model(col('value')))
print("type of predictinos >> ", type(predictions))
print("-----------prediction success----------------")

# predictions = df.withColumn('predictions', loaded_model(struct(*map(col, df.columns))))
# predictions.printSchema()
# print(predictions)
# df = predictions.toPandas()
# print(df)



# loaded_model = mlflow.pyfunc.load_model(model_uri)
# predictions = loaded_model.predict(test_df)
# print(predictions)

# predictions = spark.createDataFrame(predictions)
# print("this is prediction type >> ", type(predictions))



# result variable(maybe a dataframe) . write stream / write . output mode . format . start
# query = predictions \
#     .writeStream\
#     .outputMode("append") \
#     .format("console") \
#     .start()
#     # .option("kafka.bootstrap.servers", "172.20.0.5:29092") \
#     # .option("topic", "receive-topic") \
#     # .option("checkpointLocation", "/tmp/checkpoints") \
# query.awaitTermination()

query = predictions\
    .writeStream \
    .outputMode("update") \
    .format("console") \
    .option("truncate", False) \
    .start()
query.awaitTermination()

spark.stop()
