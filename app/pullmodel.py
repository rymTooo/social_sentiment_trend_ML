import mlflow
import mlflow.pyfunc
import os
import pandas as pd

#set environmental variables
os.environ['AWS_ACCESS_KEY_ID'] = 'CWRUnvN2zh8rqE7pidsw'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'V8RfUWQlnB4QUa7rGHbvfHjhjLiOutRa8AZ9TPvy'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'

# Set the tracking URI to your MLflow server
mlflow.set_tracking_uri("http://localhost:5000")  # Replace with your MLflow server URI

# Specify the model URI based on Name and Version
model_name = "NB_model"
model_version = 1
model_uri = f"models:/{model_name}/{model_version}"  # Replace with your model name and version

#load dependencies
# dependencies = mlflow.pyfunc.get_model_dependencies(model_uri)
# print("this is dependencies ", dependencies) # dependencies return with this method will be a "string" of path to requirements.txt
# subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", dependencies])

# Load the model
loaded_model = mlflow.pyfunc.load_model(model_uri)

#load data
data =pd.DataFrame(
    {
        'text':["ohm is a good :) person #gohome", 
             "I want to go home early sad #gohome", 
             "I bet this one is negative for sure #gohome"]
    }
)
# data2 = pd.read_csv("resources/samaple_data.csv", encoding='utf-8')
# data2 = data2[['text']] #note that when selecting column use double [[]] for indexing, or the result will list instead of dataframe

# print the prediction
print(loaded_model.predict(data))