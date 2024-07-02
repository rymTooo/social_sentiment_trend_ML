from kafka import KafkaProducer
import pandas as pd
from kafka import KafkaConsumer
from io import StringIO
import os
import mlflow

# mlflow related variable
os.environ['AWS_ACCESS_KEY_ID'] = 'CWRUnvN2zh8rqE7pidsw'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'V8RfUWQlnB4QUa7rGHbvfHjhjLiOutRa8AZ9TPvy'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000' #minio api ip:port
model_name = "NB_model"
model_version = 3
model_uri = f"models:/{model_name}/{model_version}"  # Replace with your model name and version

#topics
receive_topic = 'test-topic'
send_topic = 'receive-topic'


mlflow.set_tracking_uri("http://localhost:5000")
loaded_model = mlflow.pyfunc.load_model(model_uri)

# Create a Kafka producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: str(v).encode('utf-8')
) #locate kafka server by KAFKA_ADVERTISED_LISTENERS variable

# Initialize the Kafka consumer
consumer = KafkaConsumer(
    receive_topic,
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='my-group',
    value_deserializer=lambda x: StringIO(x.decode('utf-8'))
)
while True:
    try:
        # Consume messages from the topic
        for message in consumer:
            try:
                # Parse JSON message
                df = pd.read_json(message.value)
                prediction = loaded_model.predict(df)
                json_data = prediction.to_json(orient='records')

                # Send prediction to send_topic
                producer.send(send_topic, value=json_data)
                break

            except Exception as error:
                print("Error processing message:", error)
    finally:
        # Ensure the producer and consumer are properly closed
        producer.flush()
        print("successfully send message !!!!!!")


