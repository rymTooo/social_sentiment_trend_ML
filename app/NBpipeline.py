import pandas as pd
from sklearn.model_selection import train_test_split
from naiveBayes_model import NBmodel
import os
import mlflow
from mlflow.models import infer_signature
import pickle


# this part is the environmental var for connecting mlflow to minio 
os.environ['AWS_ACCESS_KEY_ID'] = 'CWRUnvN2zh8rqE7pidsw'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'V8RfUWQlnB4QUa7rGHbvfHjhjLiOutRa8AZ9TPvy'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://s3:9000'

# this method is ONLY used for changing negative label from -1 >>> to 0
def modify_label(label):
    if label == -1:
        label = 0
    return label

dataset_path = "./sample_data.csv"
limit = 1000 # limit the training data to limit*2 size
rd_state = 10 # random state variable for train test split

dataset = pd.read_csv(dataset_path)
# dataset = pd.concat([dataset.head(limit), dataset.tail(limit)]) # reduce the size of dataset to limit*2
dataset['label'] = dataset['label'].apply(modify_label) # change negative label from -1 to 0


X=dataset[['text']] # choose the column with content
# print("this is type of X", type(X))
y=dataset['label'] # y is of type list, NOT dataframe

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=rd_state)

nb = NBmodel()
nb.fit(X_train, y_train) # train the model

predictions = nb.predict(context=None, model_input=X_train) # make prediction
scores = nb.score(X_test['text'], y_test)

#---------------------------------------mlflow part-------------------------------------------
# mlflow config
tracking_uri = "http://tracking_server:5000" #use mlflow docker ip address
experiment_name = "NaiveBayes Sentiment analysis"
run_name = "nb model"
artifact_path = "NB_sentiment_analysis_model"

#sample for mlflow model signature
sample_input = pd.DataFrame({"text":["this is a sample positive tweet", "and this is negative :("]})
sample_prediction = nb.predict(context=None,model_input=sample_input)


# Define the model hyperparameters
params = {
    "datasize": limit*2,
    "random_state": rd_state,
    "logprior":nb.logprior,
    # "loglikelihood":nb.loglikelihood # loglikelihood is too large to store as params
}

# save these 2 weight to pickle format. Will later save to model artifacts
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

    # Log the hyperparameters >> for storing model information
    mlflow.log_params(params)
    print("log param success")

    # Log the loss metric >> for model evaluation
    mlflow.log_metrics(scores)
    print("log metrics success")

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Naive Bayes for sentiment analysis")

    # Infer the model signature
    signature = infer_signature(model_input=sample_input, model_output=sample_prediction)
    print("finish creating signature")

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