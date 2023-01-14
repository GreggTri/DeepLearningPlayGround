import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import torchtext
from tqdm import tqdm
import EmbeddedLSTM
import time
from TrainPipline import train
from CustomDataset import searchData, MyCollate

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("running on the GPU")
else: 
    device = torch.device("cpu")
    print("running on the CPU")

#test NN function
def test(model, test_dataset, criterion, device):
    model.eval()  # set the model to evaluation mode
    with torch.no_grad():  # disable gradient calculation
        total_loss = 0
        total_correct = 0
        total_samples = 0
        for inputs, labels in test_dataset:
            #inputs = inputs.to(device)  # move input and label tensors to the device
            #labels = labels.to(device)
            outputs = model(inputs)  # forward pass
            
            threshold = 0.1

            # Create a mask of the output tensor where elements greater than the threshold are set to 1 and others are set to 0
            mask = (outputs >= threshold).float()

            # Get the predicted labels by applying the mask to the output tensor
            prediction = mask.long()
        
            #print("this is prediction",prediction)
            
            loss = criterion(outputs, labels)  # compute the loss
            total_loss += loss.item()  # update the total loss
            #_, predicted = torch.max(outputs.data, 1)  # get the predicted class
            total_correct += (prediction == labels).sum().item()  # update the total number of correct predictions
            total_samples += labels.size(0)
        return total_loss / total_samples, total_correct / total_samples

dataset = searchData('Data/AUGMQdata.csv')

batch_size = 12

train_size = int(.9 * len(dataset))
eval_size = int(0.05 * len(dataset))
test_size = len(dataset) - train_size - eval_size
train_dataset, eval_dataset, test_dataset = random_split(dataset, [train_size, eval_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=MyCollate())
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=MyCollate())
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=MyCollate())
        
                
glove = torchtext.vocab.GloVe(name="6B", dim=100)
glove.stoi.__setitem__('<PAD>', 0)
glove.stoi.__setitem__('<UNK>', 1)

MODEL_NAME = f"model-{int(time.time())}"

net = EmbeddedLSTM.Net(
        len(glove.itos), #vocab_size
        embedding_dim=100,
        hidden_dim=128,
        output_dim=5
        )#.to(device)

net.embedding.from_pretrained(glove.vectors, freeze=True)
criterion = nn.MultiLabelSoftMarginLoss()

train(
    net, #Network
    train_dataloader, #train_data
    eval_dataloader, #eval_data
    18, # EPOCHS
    batch_size, #Batch Size
    device,
    criterion,
    MODEL_NAME,
    0.001 #Learning Rate
    )

test_loss, test_acc = test(net, test_dataloader, criterion, device)
print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')