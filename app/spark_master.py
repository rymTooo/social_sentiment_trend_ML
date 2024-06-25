from pyspark.sql import SparkSession
import mlflow 
import os
import pandas as pd

# might need to change localhost to something else
spark = SparkSession \
    .builder \
    .appName("StructuredNetworkWordCount") \
    .config("spark.driver.host", "localhost")\
    .getOrCreate()

#**** may use read instead of readstream ****
df = spark \
  .readStream \
  .format("kafka") \
  .option("kafka.bootstrap.servers", "kafka:9092") \
  .option("subscribe", "demo") \
  .load()

df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)")
df = pd.DataFrame(df)
print("this is df columns : ", df.columns)


os.environ['AWS_ACCESS_KEY_ID'] = 'CWRUnvN2zh8rqE7pidsw'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'V8RfUWQlnB4QUa7rGHbvfHjhjLiOutRa8AZ9TPvy'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'

model_name = "NB_model"
model_version = 2
model_uri = f"models:/{model_name}/{model_version}"

mlflow.set_tracking_uri("http://localhost:5000")

loaded_model = mlflow.spark.load_model(model_uri, spark=spark)


# Predict using the loaded model
predictions = loaded_model.predict(df) #????????????
predictions.show()

spark.stop()

#result variable(maybe a dataframe) . write stream / write . output mode . format . start
query = predictions \
    .writeStream \
    .outputMode("update") \
    .format("console") \
    .start()

query.awaitTermination()