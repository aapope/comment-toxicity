import torch.nn as nn
import torch.nn.functional as F

from utils import PADDING_IDX


class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, num_lstm_layers,
                 hidden_dims, vocab_size):
        super(LSTMClassifier, self).__init__()

        self.hidden_dims = hidden_dims

        self.embedding = nn.Embedding(vocab_size, embedding_dim,
                                      padding_idx=PADDING_IDX)
        self.lstm = nn.LSTM(embedding_dim, hidden_dims[0], num_lstm_layers)

        for i in range(1, len(hidden_dims)):
            setattr(self, f'dense_{i}', nn.Linear(in_features=hidden_dims[i-1],
                                                  out_features=hidden_dims[i]))
        self.final_dense = nn.Linear(in_features=hidden_dims[-1], out_features=6)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.t()
        lengths = x[0, :]
        texts = x[1:, :]

        embeds = self.embedding(texts)
        out, _ = self.lstm(embeds)

        for i in range(1, len(self.hidden_dims)):
            out = F.relu(getattr(self, f'dense_{i}')(out))
        out = self.final_dense(out)
        
        out = out[lengths - 1, range(len(lengths))]
        return self.sig(out)
        
