import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
        

    def forward(self, x):
        x = self.embedding(x)
        #print(x.shape)
        x, _ = self.lstm(x)
        x, _ = self.lstm2(x) #[batch_size, seq_len, hidden_dim]
        #print(x.shape)
        x = x.reshape(x.shape[0], x.shape[1]*self.hidden_dim ) #[batch_size, seq_len * hidden_dim] ex: [32, 1920]

        linear = nn.Linear(x.shape[1], self.hidden_dim) #reduces (seq_len * hidden_dim) to (hidden_dim)

        x = linear(x) #[batch_size, hidden_dim] [32, 128]
        x = self.fc1(x)

        return x