import subprocess
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import struct, col
import mlflow 
import os
import pandas
from naiveBayes_model import NBmodel 
import nltk
from pyspark.sql.functions import from_json
from pyspark.sql.types import StructType, StringType
import json


# might need to change localhost to something else
spark = SparkSession \
    .builder \
    .master("local") \
    .appName("StructuredNetworkWordCount") \
    .config("spark.driver.host", "spark-master")\
    .getOrCreate()

nltk.download('stopwords')

os.environ['AWS_ACCESS_KEY_ID'] = 'CWRUnvN2zh8rqE7pidsw'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'V8RfUWQlnB4QUa7rGHbvfHjhjLiOutRa8AZ9TPvy'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://172.20.0.2:9000' #minio api ip:port

# Set the tracking URI to your MLflow server
mlflow.set_tracking_uri("http://172.20.0.3:5000")  # Replace with your MLflow server URI

# Specify the model URI using model registry
model_name = "NB_model"
model_version = 2
model_uri = f"models:/{model_name}/{model_version}"  # Replace with your model name and version
model_path = "./naiveBayes_model.py"

loaded_model = mlflow.pyfunc.load_model(model_uri)

"""
# alternative method for loading model using minio client
client = minio.Minio(
    "localhost:9000",
    access_key="CWRUnvN2zh8rqE7pidsw",
    secret_key="V8RfUWQlnB4QUa7rGHbvfHjhjLiOutRa8AZ9TPvy",
    secure=False  # Set to True if using HTTPS
)
bucket_name = "mlflow"
object_name = "1/4c11f07b46654cda88b3840d9f655732/artifacts/NB_sentiment_analysis_model/python_model.pkl"
download_path = "/model/python_model.pkl"
client.fget_object(bucket_name, object_name, download_path)
print(f"File '{object_name}' downloaded successfully to '{download_path}'.")


"""


def process_row(row):
    if row == None or len(row) <= 0:
        print(row)
        print("exit ---------------------")
        return 0
    df = pandas.DataFrame(json.loads(row[0]['value_string']))  
    print(df)
    predictions = loaded_model.predict(df)
    print(predictions)
    return predictions


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

# query = df\
#     .writeStream \
#     .foreach(process_row) \
#     .start()
    # .option("update") \
    # .format("console") \
# query = df.writeStream.foreachBatch(
#     lambda df, epoch_id: process_row(df.collect())
# ).option("topic", kafka_topic) \
# .start()


query = df.writeStream.foreachBatch(
    lambda df, epoch_id: process_row(df.collect())
).start()
query.awaitTermination()
spark.stop()
