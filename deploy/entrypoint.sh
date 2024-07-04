#!/bin/bash

/etc/confluent/docker/run &

echo "Waiting for Kafka to be ready..."
while ! nc -z kafka 29092; do
  sleep 1
done

kafka-topics --create --topic raw-data-topic --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
kafka-topics --create --topic prediction-topic --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1

wait