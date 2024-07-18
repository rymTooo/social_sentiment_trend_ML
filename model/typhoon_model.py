import torch
from transformers import pipeline, Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
import mlflow
import os
from mlflow.models import infer_signature
import pickle
from dotenv import load_dotenv
import pandas as pd
# from tokenizer import Thai_tokenizer
import requests
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import json
from langchain_core.prompts import ChatPromptTemplate

class Typhoon_model:
    def __init__(self, model_name="typhoon-v1.5-instruct",num_labels=3):
        self.model_name = model_name
        self.num_labels = num_labels
        # self.tokenizer = AutoTokenizer.from_pretrained("scb10x/typhoon-7b")
        # # self.model = AutoModelForCausalLM.from_pretrained("scb10x/typhoon-7b")
        # self.pipeline = pipeline("sentiment-analysis", model=model_name, tokenizer=self.tokenizer)
        # messages = [
        #     {"role": "user", "content": "Who are you?"},
        # ]
        # pipe = pipeline("text-generation", model="scb10x/typhoon-v1.5-72b")
        # pipe(messages)


    def make_request(self, message):
        template = ChatPromptTemplate.from_messages([
            ("system", ""),
            ("system", "Task : Your task is to assist with  answer all question and give all detailed about it."),
            ("system", "Context: You have access to all of knowledge on all thing."),
            # ("system", "Format: Answer Thai languague only"),
            ("human", "{user_input}"),
            ("ai", "Response:"),
        ])
        endpoint = 'https://api.opentyphoon.ai/v1/chat/completions'
        res = requests.post(endpoint, json={
            "model": "typhoon-v1.5-instruct",
            "max_tokens": 512,
            "messages": message,
            "temperature": 0.4,
            "top_p": 0.9,
            "top_k": 0,
            "repetition_penalty": 1.05,
            "min_p": 0.05
        }, headers={
            "Authorization": "Bearer sk-ATxbxvWc2jsDEnUt4p19nzPFXzQKNLnp2PB9RNLjck9htbW4",
        })
        return res
    

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)

    def predict(self, input_text):
        # prediction = self.pipeline(input_text)[0]
        # print(prediction)
        # if prediction['score'] <= 0.8:
        #     prediction['label'] = 'NEUTRAL'

        prediction = self.make_request(input_text, self.url, self.api_key)
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

    def call_llm(self, message: str):
        url = "http://192.168.123.110:11434/api/generate"
        """
        Call ollama API
        """
        print(f"Call LLM with message: [{message}]")
        payload = json.dumps({
        # "model": "llama3",
        "model": "llama3-typhoon-8b:latest",
        "prompt": message,
        "stream": False
        })
        headers = {
        'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        response_json = json.loads(response.text)
        return response_json['response']
        
# test_sentence = """โครงสร้างเศรษฐกิจของจังหวัดเชียงรายมาจากการเกษตร ป่าไม้ และการประมงเป็นหลัก 
# พืชสำคัญทางเศรษฐกิจของจังหวัดเชียงราย ได้แก่ ข้าวจ้าว ข้าวโพดเลี้ยงสัตว์ สัปปะรด มันสัมปะหลัง 
# ส้มโอ ลำไย และลิ้นจี่ ซึ่งทั้งคู่เป็นผลไม้สำคัญที่สามารถปลูกได้ในทุกอำเภอของจังหวัด"""
messages = [
        {
            "role": "system",
            "content": """You are a good assistant named Typhoon.
                 Your answer can either be 'good' or 'bad' or 'neutral' only.
                 You will choose your answer based on the sentiment of the input from user.
                 You will also give the confidence score of your answer in the format of 'answer : [sentiment] , confident score : [score]'.
                 The confident score will be between 0 and 1. 
                 1 being the most confident and 0 being the least confident."""
        },
        {
            "role": "user",
            "content": "ขอสูตรไก่ย่าง"
        }
        ]
template = [
        {
            "role": "system",
            "content": "You are a helpful assistant. You will give the answer in the format of { 'sentiment' : 'good', 'confident_score' : '0.9'}. the sentiment will be either 'good', 'bad', or 'neutral' and confident_score will be between 1 and 0. "
        },
        {
            "role": "user",
            "content": "ขอสูตรไก่ย่าง"
        },
        {
            "role": "assistant",
            "content": "{ 'sentiment' : 'good', 'confident_score' : '0.9'}"
        },
        {
            "role": "user",
            "content": "โจรวิ่งหนีตำรวจลงคลอง"
        },
        {
            "role": "assistant",
            "content": "{ 'sentiment' : 'bad', 'confident_score' : '0.9'}"
        },
        {
            "role": "user",
            "content": "ป่าไม้ในประเทศไทย"
        },
        {
            "role": "assistant",
            "content": "{ 'sentiment' : 'neutral', 'confident_score' : '1'}"
        },
        {
            "role":"user",
            "content":"ผู้ชายคนหนึ่งเดินเข้าป่าลึก"
        }
        ]
typhoon = Typhoon_model()
response = typhoon.make_request(template)
print(json.loads(response.text)['choices'][0]['message']) #
# {"id":"8a4934fbcde9a8e3-SIN","choices":[{"finish_reason":"stop","index":0,"message":{"content":"{ 'sentiment' : 'neutral', 'confident_score' : '0.8'}","role":"assistant"}}],"created":1721208560,"model":"typhoon-v1.5-instruct","object":"chat.completion","system_fingerprint":null,"usage":{"completion_tokens":21,"prompt_tokens":162,"total_tokens":183}}




"""
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
