#!/bin/bash

# Start Kafka in the background
/etc/confluent/docker/run &

# Wait for Kafka to be ready
echo "Waiting for Kafka to be ready..."
while ! nc -z localhost 9092; do
  sleep 1
done

# Create a topic
kafka-topics --create --topic test-topic --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1

kafka-topics --create --topic receive-topic --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1

# Keep the container running
wait
