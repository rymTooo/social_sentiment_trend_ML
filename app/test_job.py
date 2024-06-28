
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import StringType
from pyspark.sql.functions import struct, col
import mlflow 
import os
import pandas as pd
import minio
import pickle
import joblib

# might need to change localhost to something else
spark = SparkSession \
    .builder \
    .appName("StructuredNetworkWordCount") \
    .config("spark.driver.host", "spark-master")\
    .getOrCreate()

# df = pd.DataFrame(df)
data = {'text':["this is a sample positive tweet", "and this is negative :("]}
df = pd.DataFrame(data)

spark_df = spark.createDataFrame(df)
print("this is df columns : ", spark_df.columns)
print("this is df type >>>>>>>>>", type(spark_df))

# Show DataFrame content
spark_df.show(truncate=False)

# Display DataFrame content
spark_df.show()


# os.environ['AWS_ACCESS_KEY_ID'] = 'CWRUnvN2zh8rqE7pidsw'
# os.environ['AWS_SECRET_ACCESS_KEY'] = 'V8RfUWQlnB4QUa7rGHbvfHjhjLiOutRa8AZ9TPvy'
# os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://172.20.0.2:9000' #'http://172.20.0.2:9000' # http://localhost:9000

# # Set the tracking URI to your MLflow server
# mlflow.set_tracking_uri("http://172.20.0.3:5000")  # Replace with your MLflow server URI

client = minio.Minio(
    "172.20.0.2:9000",
    access_key="CWRUnvN2zh8rqE7pidsw",
    secret_key="V8RfUWQlnB4QUa7rGHbvfHjhjLiOutRa8AZ9TPvy",
    secure=False  # Set to True if using HTTPS
)
bucket_name = "mlflow"
object_name = "1/4c11f07b46654cda88b3840d9f655732/artifacts/NB_sentiment_analysis_model/python_model.pkl"
download_path = "/model/python_model.pkl"
client.fget_object(bucket_name, object_name, download_path)
print(f"File '{object_name}' downloaded successfully to '{download_path}'.")

loaded_model = joblib.load(download_path)

# Specify the model URI using model registry
# model_name = "NB_model"
# model_version = 1
# model_uri = f"models:/{model_name}/{model_version}"  # Replace with your model name and version


# loaded_model = mlflow.pyfunc.spark_udf(
#     spark, 
#     model_uri="s3://mlflow/1/22d7037108404808b5942ba9ea4fe0b1/artifacts/NB_sentiment_analysis_model"
#     )
print("-----------load model success----------------")


# Predict using the loaded model
try:
    predictions = loaded_model.predict(data)
    # predictions.show()
except:
    print("prediction fail")

# try:
#     predictions2 = spark_df.withColumn('predictions', loaded_model(struct(*map(col, spark_df.columns))))
#     # predictions2.show()
# except:
#     print("prediction2 with struct fail")


spark.stop()
