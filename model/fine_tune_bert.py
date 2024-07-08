import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import pipeline

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

input_text = "technology can be beneficial and also dangerous on its own"

classifier = pipeline('sentiment-analysis', model=model,tokenizer=tokenizer)

prediction = classifier(input_text)[0]
if prediction['score'] <= 0.8:
    prediction['label'] = 'NEUTRAL'

print(prediction)


"""
Fine tuning

After all the effort of loading and preparing the data and datasets, creating the model and defining its loss and optimizer. This is probably the easier steps in the process.

Here we define a training function that trains the model on the training dataset created above, specified number of times (EPOCH), An epoch defines how many times the complete data will be passed through the network.

Following events happen in this function to fine tune the neural network:

The dataloader passes data to the model based on the batch size.
Subsequent output from the model and the actual category are compared to calculate the loss.
Loss value is used to optimize the weights of the neurons in the network.
After every 5000 steps the loss value is printed in the console.
As you can see just in 1 epoch by the final step the model was working with a miniscule loss of 0.0002485 i.e. the output is extremely close to the actual output.
"""

# # Function to calcuate the accuracy of the model

# def calcuate_accu(big_idx, targets):
#     n_correct = (big_idx==targets).sum().item()
#     return n_correct

# Defining the training function on the 80% of the dataset for tuning the distilbert model

# def train(epoch):
#     tr_loss = 0
#     n_correct = 0
#     nb_tr_steps = 0
#     nb_tr_examples = 0
#     model.train()
#     for _,data in enumerate(training_loader, 0):
#         ids = data['ids'].to(device, dtype = torch.long)
#         mask = data['mask'].to(device, dtype = torch.long)
#         targets = data['targets'].to(device, dtype = torch.long)

#         outputs = model(ids, mask)
#         loss = loss_function(outputs, targets)
#         tr_loss += loss.item()
#         big_val, big_idx = torch.max(outputs.data, dim=1)
#         n_correct += calcuate_accu(big_idx, targets)

#         nb_tr_steps += 1
#         nb_tr_examples+=targets.size(0)
        
#         if _%5000==0:
#             loss_step = tr_loss/nb_tr_steps
#             accu_step = (n_correct*100)/nb_tr_examples 
#             print(f"Training Loss per 5000 steps: {loss_step}")
#             print(f"Training Accuracy per 5000 steps: {accu_step}")

#         optimizer.zero_grad()
#         loss.backward()
#         # # When using GPU
#         optimizer.step()

#     print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
#     epoch_loss = tr_loss/nb_tr_steps
#     epoch_accu = (n_correct*100)/nb_tr_examples
#     print(f"Training Loss Epoch: {epoch_loss}")
#     print(f"Training Accuracy Epoch: {epoch_accu}")

#     return 