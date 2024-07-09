
#pip install kafka-python **NOT** >> pip install kafka
from kafka import KafkaProducer
import pandas as pd
from dotenv import load_dotenv
import os


load_dotenv()

server_ip = os.getenv('SERVER_IP')
kafka_port = os.getenv('KAFKA_EXTERNAL_PORT')

print("server port >> ", server_ip)
print("kafka port >> ", kafka_port)


# Create a Kafka producer
producer = KafkaProducer(
    bootstrap_servers=f'{server_ip}:{kafka_port}',
    value_serializer=lambda v: str(v).encode('utf-8'),
    key_serializer= lambda x: str(x).encode('utf-8')
)
#locate kafka server by KAFKA_ADVERTISED_LISTENERS variable

# Define the topic
topic = 'raw-data-topic'
df = pd.DataFrame(
    {
        "text":["this is example tweeet number 1 #first surely it is good", 
                "this is the second one bad and worse as hell", 
                "123"]
    }
)
# df = pd.read_csv("resources/sample_data.csv")[['text']]
# df = pd.concat([df.head(100), df.tail(100)])

json_data = df.to_json(orient='records')
print("this is the data we sending >> \n", json_data)
key = "tech"
producer.send(topic, value = json_data, key= key)
# Close the producer
producer.flush()
producer.close()
