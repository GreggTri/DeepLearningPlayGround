import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import torch.optim as optim
import time

def eval_model(model, eval_dataset, criterion, device):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_correct = 0
        total_samples = 0
        for inputs, labels in eval_dataset:
            #inputs = inputs.to(device)
            #labels = labels.to(device) #sends inputs and labels to the GPU
            outputs = model(inputs)#forward pass
            
            loss = criterion(outputs, labels) #computes loss
            
            total_loss += loss.item()
            
            threshold = 0.1

            # Create a mask of the output tensor where elements greater than the threshold are set to 1 and others are set to 0
            mask = (outputs >= threshold).float()

            # Get the predicted labels by applying the mask to the output tensor
            prediction = mask.long()
        
            #print("this is prediction",prediction)
            #print("this is labels",labels)
            
            total_correct += (prediction == labels).sum().item() #getting error here
            #RuntimeError: The size of tensor a (11) must match the size of tensor b (5) at non-singleton dimension 1
            total_samples += labels.size(0)
        return total_loss/ total_samples, total_correct/total_samples


def train(model, train_data, eval_dataset, EPOCHS, BATCH_SIZE, device, criterion, MODEL_NAME, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Loop over the specified number of epochs
    with open(f"{MODEL_NAME}", "a") as f:
        for epoch in range(EPOCHS):
            model.train()
            # Loop over the training data
            for inputs, labels in train_data:
                # Clear the gradients
                model.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                
                
                # Compute the loss and backpropagate the gradients
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, labels)]
                acc = matches.count(True)/len(matches)
                #end of inner loop
                
            eval_loss, eval_acc = eval_model(model, eval_dataset, criterion, device)
            
            acc_percent = 100 * acc / len(train_data)
            f.write(f"{MODEL_NAME},{round(time.time(),3)},{round(float(acc_percent),2)},{round(float(loss),4)},{round(float(eval_acc),2)},{round(float(eval_loss),4)}\n")
            #,{round(float(eval_acc),2)},{round(float(eval_loss),4)}
            
            # Print the loss at the end of each epoch
            print(f"Epoch {epoch + 1}/{EPOCHS}: training_loss = {loss:.3f} training_acc = {acc_percent:.2f}%")