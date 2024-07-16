from pyspark.sql import SparkSession

def create_spark_session(master_url):
    spark = SparkSession.builder \
        .appName("PySparkTest") \
        .master(master_url) \
        .getOrCreate()
        #.config("spark.driver.host", "spark-master") \
        #.config("spark.driver.bindAddress", "0.0.0.0") \
        #.config("spark.authenticate", "true") \
        #.config("spark.authenticate.secret", "your_secret_key") \
        
    
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def run_test(spark):
    data = [("Alice", 1), ("Bob", 2), ("Charlie", 3)]
    columns = ["Name", "Value"]
    
    df = spark.createDataFrame(data, columns)
    df.show()
    
    df_transformed = df.withColumn("Value", df["Value"] * 2)
    df_transformed.show()

def main():
    master_url = "spark://localhost:7077"
    spark = create_spark_session(master_url)
    
    try:
        run_test(spark)
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
