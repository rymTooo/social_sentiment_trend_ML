#pip install kafka-python **NOT** >> pip install kafka
from kafka import KafkaProducer
import pandas as pd

# Create a Kafka producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: str(v).encode('utf-8')
) #locate kafka server by KAFKA_ADVERTISED_LISTENERS variable

# Define the topic
topic = 'test-topic'

# Send messages to the topic
# for i in range(5):
#     message = f'Message {i}'
#     producer.send(topic, value=message.encode('utf-8')) # send message to {topic name} and {message value.encode('uft-8')}
#     print(f'Sent: {message}')
count = 1
df = pd.DataFrame({"text":["this is example tweeet number 1 #first", "this is the second one", f"{count}"]})
json_data = df.to_json(orient='records')
print(json_data)
producer.send(topic, value = json_data)


# Close the producer
producer.flush()
producer.close()
