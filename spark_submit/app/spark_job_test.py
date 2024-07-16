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
import json
from dotenv import load_dotenv
from pyspark import SparkConf


# might need to change localhost to something else
spark_packages = [
    "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0",
    "org.apache.kafka:kafka-clients:3.2.1",
    "org.apache.spark:spark-tags_2.12:3.2.0",
    "org.slf4j:slf4j-api:1.7.29",
    "org.slf4j:slf4j-log4j12:1.7.29",
]
spark_conf = SparkConf()
spark_conf.set("spark.jars.packages", ",".join(spark_packages))
spark_conf.set("spark.executor.memory", "500m")
spark_conf.set("spark.driver.memory", "500m")
spark_conf.set("spark.executor.cores", "1")
spark_conf.set("spark.driver.cores", "1")
spark_conf.set("spark.memory.fraction", "0.8")
spark_conf.set("spark.cores.max", "1")
spark_conf.set("spark.executor.instances", "2")
spark_conf.set("spark.driver.host", "10.0.1.54")
spark_conf.set("spark.driver.bindAddress", "0.0.0.0")
spark_conf.set("spark.sql.streaming.forceDeleteTempCheckpointLocation", True)

spark = (
    SparkSession.builder.master("spark://10.0.1.54:27077")
    .appName("sentiment analysis on tweet")
    .config(conf=spark_conf)
    .getOrCreate()
)

#set env
load_dotenv()
server_ip = os.getenv('SERVER_IP')
mlflow_port = os.getenv('MLFLOW_PORT')
kafka_port = os.getenv('KAFKA_CLIENT_PORT')

#define kafka topic
input_topic = "test-raw-data-topic"
output_topic = "test-prediction-topic"


nltk.download('stopwords')

# os.environ['AWS_ACCESS_KEY_ID'] = 'CWRUnvN2zh8rqE7pidsw'
# os.environ['AWS_SECRET_ACCESS_KEY'] = 'V8RfUWQlnB4QUa7rGHbvfHjhjLiOutRa8AZ9TPvy'
# os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://s3:9000' #minio api ip:port
# Set the tracking URI to your MLflow server
mlflow.set_tracking_uri(f"http://{server_ip}:{mlflow_port}")  # Replace with your MLflow server URI

# Specify the model URI using model registry
model_name = os.getenv("REGISTERED_MODEL_NAME")
model_version = int(os.getenv("REGISTERED_MODEL_VERSION"))
model_uri = f"models:/{model_name}/{model_version}"  # Replace with your model name and version


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

results = []
# input format {"snippet" : "message"}

def process_row(row):
    if row == None or len(row) <= 0:
        print(row)
        print("exit ---------------------")
        return 0
    print(row)
    print('type of row[0] .. ', type(row[0]))
    print("this is row type >> ", type(row))
    print("\n\n this is this >>", row[0])
    data_dict = json.loads(row[0]['value_string'])
    df = pandas.DataFrame([data_dict])
    # df['text'] = df['snippet']
    print(df)
    predictions = loaded_model.predict(df[['text']])
    print(predictions)
    results.append({"value": predictions.to_json(orient='records')})
    
    predictions_df = spark.createDataFrame(results)
    predictions_df.printSchema()
    # Write predictions to Kafka
    predictions_df.selectExpr("CAST(value AS STRING)") \
        .write \
        .format("kafka") \
        .option("kafka.bootstrap.servers", f"{server_ip}:{kafka_port}") \
        .option("topic", output_topic) \
        .save()
    

    return predictions_df


#**** may use read instead of readstream ****
df = spark \
  .readStream \
  .format("kafka") \
  .option("kafka.bootstrap.servers", f"{server_ip}:{kafka_port}") \
  .option("subscribe", input_topic) \
  .load()
df.selectExpr("CAST(value AS STRING)")
df.printSchema()
df = df.withColumn("value_string", col("value").cast("string"))


query = df.writeStream \
    .foreachBatch(
        lambda df, epoch_id: process_row(df.collect())
    ).start()

query.awaitTermination()
spark.stop()
