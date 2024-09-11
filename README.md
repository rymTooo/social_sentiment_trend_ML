# Sentiment analysis
<p>This repository include the file necessary in the ML pipeline. The included framework are</p>

This repositoray alone doesn't represent the whole project. The project presentation slide can be found in [Download the PDF](./Social_Sentiment_Trends_Presentation.pdf)

    
- **mlflow** : platform for model management
- **minio**  : object storage for storing model artifacts
- **kafka**  : message broker for send receive data between source and spark
- **spark**  : paltform for distributed computation, used for model prediciton

## Getting started
This guide will walk you through necessary steps to set up the pipeline.
### Configuration

### Installation
Create conda env and install necessary dependencies <br>

```shell
conda create -n ml python=3.11.9

```
```shell
conda activate ml

```
```shell
pip install mlflow==2.14.1 cloudpickle==3.0.0 nltk==3.8.1 numpy==1.26.4 pandas==2.2.2 scikit-learn==1.5.0 scipy==1.13.1 minio==7.2.7 boto3==1.34.134 pyspark==3.5.1 psutil==6.0.0

```


### Quick started

Since the model data won't be in repository, you will need to train the model once. <br>

1. up docker compose for minio, mlflow, kafka, and zookeeper
    ```shell
    docker-compose -f ./deploy/docker-compose.yml up --build -d

    ```
2. Logging in to minio and mlflow browser interface
    - [Minio](http://localhost:9000)
    - [MLflow](http://localhost:5000)

3. Set **minio bucket** and **access key** <br>
    Create the bucket name **mlflow**<Br>
    Create the access key and set the configuration to:
    - Access key
    ``CWRUnvN2zh8rqE7pidsw``
    - Secret key
    ``V8RfUWQlnB4QUa7rGHbvfHjhjLiOutRa8AZ9TPvy``
4. Run model training code

    Set the **_limit_** variable in **_local_app/NBpipeline.py_**
    ```python
    dataset_path = "./resources/sample_data.csv"
    limit = 1000 # limit the training data to limit*2 size
    rd_state = 10 # random state variable for train test split
    ```
    run **NBpipeline.py**, you should see the model showing up in mlflow brower UI 
    ```shell
    python local_app/NBpipeline.py
    ```

You should now have model saved and ready to go, let start spark container and try predicting with our model.

1. Create kafka topics

    execute into kafka container
    ```shell
    docker exec -it kafka_test /bin/bash
    ```

    Create the topic
    ```shell
    kafka-topics --create --topic raw-data-topic --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1 \

    kafka-topics --create --topic prediction-topic --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
    ```

2. Run spark container using
    ```shell
    docker-compose -f ./deploy_spark/docker-compose.yml up --build -d
    ```
3. Execute in to spark container
    ```shell
    docker exec -it spark-master /bin/bash
    ```

4. run the script
    ```
    ./run.sh
    ```
    >you should now see spark console load some dependencies and it should be ready to receive and process incoming data

Let's try send data to spark and see if it return the result

1. Run ***read_message.py*** locally
    ```shell
    python local_app/read_message.py
    ```

2. Open another terminal and Run ***send_message.py*** 
    ```shell
    python local_app/send_message.py
    ```
> ***send_message.py*** will send data to raw-data-topic for spark to process while ***read_message.py*** will continously read any message from prediction-topic which spark will write prediction result to.

3. Check the result in spark terminal, it should show the prediction on data from the model.
