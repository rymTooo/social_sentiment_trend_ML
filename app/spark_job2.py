import subprocess
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import struct, col
import mlflow 
import os
import pandas
import nltk
from pyspark.sql.functions import from_json
from pyspark.sql.types import StructType, StringType
import json


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
df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)")
df = df.withColumn("value_string", col("value").cast("string"))
df.printSchema()




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

loaded_model = mlflow.pyfunc.load_model(model_uri)



def process_row(row):
    print("this is type of the row obj >>>>>>>", type(row))
    print(">>>>>>>>>" , row['value_string'])
    df = pandas.DataFrame(json.loads(row['value_string']))
    print(df)
    predictions = loaded_model.predict(df)
    print(predictions)


query = df\
    .writeStream \
    .foreach(process_row) \
    .start()
    # .option("update") \
    # .format("console") \
query.awaitTermination()
spark.stop()
