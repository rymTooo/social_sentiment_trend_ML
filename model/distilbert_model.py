import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import pipeline, Trainer, TrainingArguments
import mlflow
import os
from mlflow.models import infer_signature
import pickle
from dotenv import load_dotenv
import pandas as pd


class Distilbert_model:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english", num_labels=3):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.pipeline = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)

    def predict(self, input_text):
        prediction = self.pipeline(input_text)[0]
        if prediction['score'] <= 0.8:
            prediction['label'] = 'NEUTRAL'
        return prediction

    def train(self, train_dataset, test_dataset, output_dir="./results", epochs=3, batch_size=16, learning_rate=1e-5):
        # Tokenize datasets
        tokenized_train = train_dataset.map(self.tokenize_function, batched=True)
        tokenized_test = test_dataset.map(self.tokenize_function, batched=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            tokenizer=self.tokenizer
        )

        trainer.train()

    def evaluate(self, eval_dataset):
        # Tokenize dataset
        tokenized_eval = eval_dataset.map(self.tokenize_function, batched=True)
        
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            per_device_eval_batch_size=16,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            eval_dataset=tokenized_eval,
            tokenizer=self.tokenizer
        )

        metrics = trainer.evaluate()
        return metrics
#load env
load_dotenv()
server_ip = os.getenv('SERVER_IP')
mlflow_port = os.getenv('MLFLOW_PORT')


input_text = "technology can be beneficial and also dangerous on its own"



#---------------------------------------mlflow part-------------------------------------------
# mlflow config
tracking_uri = f"http://{server_ip}:{mlflow_port}" #use mlflow docker ip address
experiment_name = os.getenv('EXPERIMENT_NAME')
run_name = os.getenv('RUN_NAME')
artifact_path = os.getenv('ARTIFACT_PATH')

#sample for mlflow model signature
sample_input = pd.DataFrame({"text":["this is a sample positive tweet", "and this is negative :("]})
sample_prediction = prediction


# Define the model hyperparameters
params = {
    "model": 'distilbert-base-uncased-finetuned-sst-2-english'
}


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
        registered_model_name=os.getenv("REGISTERED_MODEL_NAME"),
    )















"""
Fine tuning
https://www.kaggle.com/code/nadzmiagthomas/distilbert-fine-tuning/notebook
"""

"""
#load pretrained model

from transformers import  AutoModelForSequenceClassification, AutoTokenizer
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english" # Define which pre-trained model we will be using
classifier = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2) # Get the classifier
tokenizer = AutoTokenizer.from_pretrained(checkpoint) # Get the tokenizer

import pandas as pd
# Load the training data
train_path = '/kaggle/input/nlp-getting-started/train.csv'
df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

print(df.head())

# get the specifc columns
df = df.loc[:,["text", "target"]]

# Split the data into train and evaluation (stratified)
from sklearn.model_selection import train_test_split
df_train, df_eval = train_test_split(df, train_size=0.8,stratify=df.target, random_state=42) # Stratified splitting 


# convert pandas df to dataset
from datasets import Dataset, DatasetDict
raw_datasets = DatasetDict({
    "train": Dataset.from_pandas(df_train),
    "eval": Dataset.from_pandas(df_eval)
})

# Check the datasets
print("Dataset Dict:\n", raw_datasets)
print("\n\nTrain's features:\n", raw_datasets["train"].features)
print("\n\nFirst row of Train:\n", raw_datasets["train"][0])


#tokenized the data
tokenized_datasets = raw_datasets.map(lambda dataset: tokenizer(dataset['text'], truncation=True), batched=True)
print(tokenized_datasets)


# Check the first row
print(tokenized_datasets["train"][0])

# clean dataset *** might not be necessary ***
tokenized_datasets = tokenized_datasets.remove_columns(["text", "__index_level_0__"])
tokenized_datasets = tokenized_datasets.rename_column("target", "labels")
print(tokenized_datasets)


# !pip -q install evaluate

from transformers import DataCollatorWithPadding, TrainingArguments, Trainer
import numpy as np
import evaluate

# Padding for batch of data that will be fed into model for training
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training args 
training_args = TrainingArguments("test-trainer", num_train_epochs=1, evaluation_strategy="epoch", 
                                  weight_decay=5e-4, save_strategy="no", report_to="none")

# Metric for validation error
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc") # F1 and Accuracy
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Define trainer
trainer = Trainer(
    classifier,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["eval"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Start the fine-tuning 
trainer.train()

# EVALUATION
from sklearn.metrics import classification_report

# Make prediction on evaluation dataset
y_pred = trainer.predict(tokenized_datasets["eval"]).predictions
y_pred = np.argmax(y_pred, axis=-1)

# Get the true labels
y_true = tokenized_datasets["eval"]["labels"]
y_true = np.array(y_true)

# Print the classification report
print(classification_report(y_true, y_pred, digits=3))


#SUBMISSION ????
# Get the test data
df_test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
ids = df_test.id # Save ids
df_test = df_test.loc[:,["text"]] # Keep only text

# Turn the DataFrame into appropriate format
test_dataset = Dataset.from_pandas(df_test)
test_dataset = test_dataset.map(lambda dataset: tokenizer(dataset['text'], truncation=True), batched=True)
test_dataset = test_dataset.remove_columns('text')

# Get the prediction
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=-1)

# Turn submission into DataFrame and save into CSV files
submission = pd.DataFrame({"id":ids, "target":preds})
submission.to_csv("submission.csv", index=False)
"""
