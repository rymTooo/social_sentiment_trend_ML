from kafka import KafkaConsumer
import pandas as pd
from io import StringIO

# Initialize the Kafka consumer
consumer = KafkaConsumer(
    'receive-topic',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='my-group',
    value_deserializer=lambda x: StringIO(x.decode('utf-8'))
)

# Consume messages from the topic
for message in consumer:
    try:
        df = pd.read_json(message.value)
        print(df)
    except Exception as error:
        print(">>>", error)

    print(type(message.value))
    print(f"Received message: {message.value}")