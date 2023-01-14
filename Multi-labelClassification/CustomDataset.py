import torch
import csv
import os
import pandas as pd
import spacy
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torchtext

nlp = spacy.load("en_core_web_sm")
glove = torchtext.vocab.GloVe(name="6B", dim=100)
glove.stoi.__setitem__('<PAD>', 0)
glove.stoi.__setitem__('<UNK>', 1)

class MyCollate:
    def __init__(self):
        pass

    def __call__(self, batch):
        targets = [item[0] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=0)
        new_targets = targets.transpose(0, 1)
        
        labels = [item[1] for item in batch]
        batch_size = len(batch)
        num_labels = len(labels[0]) 
        
        labels_tensor = torch.ones(batch_size, num_labels)
        
        for i, label in enumerate(labels):
            for j, (key, value) in enumerate(label.items()):        
                    labels_tensor[i, j] = value
        
        return new_targets, labels_tensor

class searchData(Dataset):
    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(path)
        self.sentence = self.df["sentence"]
        self.query = self.df["query0"]
        self.query = self.df["query1"]
        self.query = self.df["query2"]
        self.query = self.df["query3"]
        self.color = self.df["color"]
        self.vocab = glove.stoi
        self.df['labels'] = self.df.apply(self.combine_labels, axis=1)
        self.labels = self.df['labels']
        
    def combine_labels(self, row):
        colorRow = row['color']
        
        #for query0 row
        if pd.isnull(row['query0']):
            q0 = self.vocab['<UNK>']
        else:
            token_q0 = [token for token in self.tokenizer_eng(row['query0'])]
            q0 = self.vocab[token_q0[0]]

        #for query1 row
        if pd.isnull(row['query1']):
            q1 = self.vocab['<UNK>']
        else:
            token_q1 = [token for token in self.tokenizer_eng(row['query1'])]
            q1 = self.vocab[token_q1[0]]
        
        #for query2 row
        if pd.isnull(row['query2']):
            q2 = self.vocab['<UNK>']
        else:
            token_q2 = [token for token in self.tokenizer_eng(row['query2'])]
            q2 = self.vocab[token_q2[0]]
        
        #for query3 row
        if pd.isnull(row['query3']):
            q3 = self.vocab['<UNK>']
        else:
            token_q3 = [token for token in self.tokenizer_eng(row['query3'])]
            q3 = self.vocab[token_q3[0]]

        #for color row
        if pd.isnull(colorRow):
            color = self.vocab['<UNK>']
        else:
            token_color = [token for token in self.tokenizer_eng(colorRow)]
            color = self.vocab[token_color[0]]
        
        #returned value
        #return {'query': torch.tensor(query), 'price': torch.tensor(price), 'color': torch.tensor(color)}
        return {'query0': q0,'query1': q1, 'query2': q2,'query3': q3, 'color': color}
    
    @staticmethod
    def tokenizer_eng(text):
        #print(text)
        return [tok.text.lower() for tok in nlp.tokenizer(text)]
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        sentence = self.sentence[index]
        labels = self.labels[index]
        tokenized_text = []
        numericalized_text = []
        
        for word in self.tokenizer_eng(sentence):    
            tokenized_text.append(word)
        #print(tokenized_text)
        
        for token in tokenized_text:
            numericalized_text.append(self.vocab[token] if token in self.vocab.keys() else self.vocab['<UNK>']) 
        
        #print("this is a numerical text", numericalized_text)
        #("this is labels", labels)
        return (torch.tensor(numericalized_text), labels)
    
#dataset = searchData('Data/dataset.csv')
#dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

#for batch in dataloader:
#    data, label = batch
#    print(data)
#    print(label)
#    break