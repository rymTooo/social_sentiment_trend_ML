import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

inputs = tokenizer("neurtral", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

print(logits)
predicted_class_id = logits.argmax().item()
print(predicted_class_id)
model.config.id2label[predicted_class_id]