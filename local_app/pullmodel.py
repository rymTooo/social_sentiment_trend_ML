import mlflow
import mlflow.pyfunc
import os
import pandas as pd

os.environ['AWS_ACCESS_KEY_ID'] = 'CWRUnvN2zh8rqE7pidsw'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'V8RfUWQlnB4QUa7rGHbvfHjhjLiOutRa8AZ9TPvy'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://172.20.0.2:9000' #'http://172.20.0.2:9000' # http://localhost:9000

# Set the tracking URI to your MLflow server
mlflow.set_tracking_uri("http://172.20.0.3:5000")  # Replace with your MLflow server URI

# Specify the model URI using model registry
model_name = "NB_model"
model_version = 1
model_uri = f"models:/{model_name}/{model_version}"  # Replace with your model name and version

#load dependencies
# dependencies = mlflow.pyfunc.get_model_dependencies(model_uri)
# print("type ", type(dependencies))
# print("this is dependencies ", dependencies)
# subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", dependencies])
# Load the model
loaded_model = mlflow.pyfunc.load_model(model_uri)

# Use the model for predictions

data2 = pd.read_csv("samaple_data.csv", encoding='utf-8')
data2 = data2[['text']]
# data =pd.DataFrame({'text':["ohm is a good :) person #gohome", "I want to go home early sad #gohome", "I bet this one is negative for sure #gohome"]})

print(loaded_model.predict(data2))



