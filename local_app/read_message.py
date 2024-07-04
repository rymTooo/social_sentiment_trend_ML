from kafka import KafkaConsumer
import pandas as pd
from io import StringIO
from dotenv import load_dotenv
import os

load_dotenv()

topic_to_read = "raw-data-topic"
server_ip = os.getenv('SERVER_IP')
kafka_port = os.getenv('KAFKA_EXTERNAL_PORT')

print("server port >> ", server_ip)
print("kafka port >> ", kafka_port)

# Initialize the Kafka consumer
consumer = KafkaConsumer(
    topic_to_read,
    bootstrap_servers=f'{server_ip}:{kafka_port}',
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='my-group',
    value_deserializer=lambda x: StringIO(x.decode('utf-8'))
)

# Consume messages from the topic
for message in consumer:
    try:
        df = pd.read_json(message.value)
        df['hashtag'] = message.key.decode('utf-8')
        df.to_csv("resources/test.csv")
        print(df)
    except Exception as error:
        print(">>>", error)

    print(type(message.value))
    print(f"Received message: {message.value}")