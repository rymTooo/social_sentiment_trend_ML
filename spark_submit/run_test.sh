spark-submit --master spark://localhost:27077 \
--deploy-mode client \
--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.1 \
--repositories https://repo1.maven.org/maven2/ \
--files /opt/spark-job/.env \
/opt/spark-job/spark_job_test.py
