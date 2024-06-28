import pandas as pd
from sklearn.model_selection import train_test_split
from naiveBayes_model import NBmodel
import os
import mlflow
from mlflow.models import infer_signature
import pickle


# this part is the environmental var for using with mlflow-minio on docker
os.environ['AWS_ACCESS_KEY_ID'] = 'CWRUnvN2zh8rqE7pidsw'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'V8RfUWQlnB4QUa7rGHbvfHjhjLiOutRa8AZ9TPvy'
# os.environ['MLFLOW_DEFAULT_ARTIFACT_ROOT'] = 'http://localhost://mlflow'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'

def modify_label(label):
    if label == -1:
        label = 0
    return label

dataset_path = "D:/Kafka_spark/resources/sample_data.csv"
limit = 1000
rd_state = 10

dataset = pd.read_csv(dataset_path)
dataset = pd.concat([dataset.head(limit), dataset.tail(limit)])
dataset['label'] = dataset['label'].apply(modify_label)


X=dataset[['text']]
# print("this is type of X", type(X))
y=dataset['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=rd_state)

nb = NBmodel()
nb.fit(X_train, y_train)

prediction = nb.predict(context=None, model_input=X_train)
scores = nb.score(X_test['text'], y_test)

#test result on sample data
data2 = pd.read_csv("D:/Sentiment_analysis_model/resources/sample_data.csv", encoding='utf-8')
data2 = data2[['text']]
print("prediction : ", nb.predict(context=None, model_input=data2))
# prediction :  {'positive': 0.31885, 'neutral': 0.1471, 'negative': 0.53405}

print("fit success")

#sample for mlflow model signature
sample_input = pd.DataFrame({"text":["this is a sample positive tweet", "and this is negative :("]})
sample_prediction = nb.predict(context=None,model_input=sample_input)


# mlflow config
tracking_uri = "http://localhost:5000" #use mlflow docker ip address
experiment_name = "NaiveBayes Sentiment analysis"
run_name = "nb model"
artifact_path = "NB_sentiment_analysis_model"


# Define the model hyperparameters
params = {
    "datasize": limit*2,
    "random_state": rd_state,
    "logprior":nb.logprior,
    # "loglikelihood":nb.loglikelihood >>> loglikelihood
}
# print(nb.loglikelihood)

with open("logprior.pkl", "wb") as f:
    pickle.dump(nb.logprior, f)

with open("loglikelihood.pkl", "wb") as f:
    pickle.dump(nb.loglikelihood, f)

mlflow.set_tracking_uri(uri=tracking_uri)

# Create a new MLflow Experiment
mlflow.set_experiment(experiment_name=experiment_name) #name of the experiment

# Start an MLflow run
with mlflow.start_run(run_name=run_name):

    print("trakcing > ", mlflow.get_tracking_uri())
    print("artifact > ", mlflow.get_artifact_uri())

    # Log the hyperparameters
    mlflow.log_params(params)
    print("log param success")

    # Log the loss metric
    mlflow.log_metrics(scores)
    print("log metrics success")

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Naive Bayes for sentiment analysis")

    # Infer the model signature
    signature = infer_signature(model_input=sample_input, model_output=sample_prediction) # to be fix in a more specific format
    print("finish creating signature")

    # with open("save_model/python_model.pkl", 'wb') as file:
    #     pickle.dump(nb, file)

    model_info = mlflow.pyfunc.log_model(
        python_model = NBmodel(),
        artifacts={
            "logprior": "logprior.pkl",
            "loglikelihood": "loglikelihood.pkl"
        },
        artifact_path= artifact_path, #artifact folder name 
        signature=signature,
        input_example=sample_input,
        registered_model_name="NB_model",
    )

