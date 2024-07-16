from pyspark.sql import SparkSession
from pyspark import SparkConf
import pandas as pd
from pyspark.pandas.mlflow import load_model
from dotenv import load_dotenv
import os
import mlflow
import nltk
import json
from pyspark.sql.functions import col


# class SparkSubmit():
server_ip = ""
kafka_port = ""
mlflow_port = ""
spark_master_port = ""
input_topic = "test-raw-data-topic"
output_topic = "test-prediction-topic"

def create_spark_session(master_url):
    spark_conf = create_config()
    spark = SparkSession.builder \
        .appName("PySparkTest") \
        .master(master_url) \
        .config(conf=spark_conf) \
        .getOrCreate()
        #.config("spark.driver.host", "spark-master") \
        #.config("spark.driver.bindAddress", "0.0.0.0") \
        #.config("spark.authenticate", "true") \
        #.config("spark.authenticate.secret", "your_secret_key") \
        
    
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def run_test(spark):
    # load_env()
    print(server_ip)
    print(mlflow_port)
    print(kafka_port)
    print(spark_master_port)

    #define kafka topic
    

    nltk.download('stopwords')

    # # Set the tracking URI to your MLflow server
    mlflow.set_tracking_uri(f"http://{server_ip}:{mlflow_port}")  # Replace with your MLflow server URI

    # Specify the model URI using model registry
    model_name = os.getenv("REGISTERED_MODEL_NAME")
    model_version = int(os.getenv("REGISTERED_MODEL_VERSION"))
    model_uri = f"models:/{model_name}/{model_version}"  # Replace with your model name and version

    # loaded_model = mlflow.pyfunc.load_model(model_uri)
    loaded_model = load_model(model_uri)

    df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", f"{server_ip}:{kafka_port}") \
    .option("subscribe", input_topic) \
    .load()
    df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)")
    df = df.withColumn("value_string", col("value").cast("string"))
    df.printSchema()


    query = df.writeStream \
        .foreachBatch(
            lambda df, epoch_id: process_row(df.collect(), loaded_model, spark)
        ).start()

    query.awaitTermination()


def process_row(row,loaded_model,spark):
    results = []
    if row == None or len(row) <= 0:
        print(row)
        print("exit ---------------------")
        return 0
    try:
        data_dict = json.loads(row[0]['value_string'])
        df = pd.DataFrame([data_dict])
    except Exception as error:
        return error
    predictions = loaded_model.predict(df[['text']])
    print(predictions)
    # results.append({"text": predictions['text'], "label": predictions['label']})
    results.append({"value": predictions.to_json(orient='records')})
    
    predictions_df = spark.createDataFrame(results)
    predictions_df.printSchema()
    # Write predictions to Kafka
    predictions_df.selectExpr("CAST(value AS STRING)", "CAST(key AS STRING)") \
        .write \
        .format("kafka") \
        .option("kafka.bootstrap.servers", f"{server_ip}:{kafka_port}") \
        .option("topic", output_topic) \
        .save()
    

    return 0 #  predictions_df




def create_config():
    spark_packages = [
        "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1",
        "org.apache.kafka:kafka-clients:3.2.1",
        # "org.apache.spark:spark-tags_2.12:3.2.0",
        # "org.slf4j:slf4j-api:1.7.29",
        # "org.slf4j:slf4j-log4j12:1.7.29",
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
    spark_conf.set("spark.driver.host", f"{server_ip}")
    spark_conf.set("spark.driver.bindAddress", "0.0.0.0")
    spark_conf.set("spark.sql.streaming.forceDeleteTempCheckpointLocation", True)

    return spark_conf

def load_env():
    load_dotenv()
    global server_ip
    global mlflow_port
    global kafka_port
    global spark_master_port
    server_ip = os.getenv('SERVER_IP')
    mlflow_port = os.getenv('MLFLOW_PORT')
    kafka_port = os.getenv('KAFKA_CLIENT_PORT')
    spark_master_port = os.getenv('SPARK_CLUSTER_MASTER_PORT')


def main():
    load_env()
    master_url = f"spark://{server_ip}:{spark_master_port}"
    spark = create_spark_session(master_url)
    
    try:
        run_test(spark)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
