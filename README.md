# Sentiment analysi
<p>This repository include the file necessary in the ML pipeline. The included platform? framework? are</p>

    - mlflow : platform for model management
    - minio  : object storage for storing model artifacts
    - kafka  : message broker for 
    - spark  : paltform for distributed computation, used for model prediciton

## Getting started
This guide will walk you through necessary steps to set up the pipeline.
### Configuration

### Installation
Create conda env and install necessary dependencies <br>

```
conda create -n ml python=3.11.9

```
```
conda activate ml

```
```
pip install mlflow==2.14.1 cloudpickle==3.0.0 nltk==3.8.1 numpy==1.26.4 pandas==2.2.2 scikit-learn==1.5.0 scipy==1.13.1 minio==7.2.7 boto3==1.34.134 pyspark==3.5.1 psutil==6.0.0

```


### Quick started

Since the model data won't be in repository, you will need to train the model once. <br>

1. up docker compose for minio, mlflow, kafka, and zookeeper
    ```
    docker-compose up --build -d

    ```
2. Logging in to minio and mlflow browser interface
    - Minio
    ```
    localhost:9000

    ```
    - MLflow
    ```
    localhost:5000

    ```
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
    run **NBpipeline.py**
5. Run spark container using
    ```
    #assuming you are in spark_deploy folder
    docker-compose up --build
    ```
6. 

