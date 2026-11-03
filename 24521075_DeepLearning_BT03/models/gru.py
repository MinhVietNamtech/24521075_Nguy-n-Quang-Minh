from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence

class GRU(nn.Module):
    def __init__(self,
                vocab_size: int, 
                hidden_size: int, 
                n_layer: int, 
                n_labels: int,
                padding_idx: int = 0,
                dropout: float = 0.2):
        super(GRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=n_layer, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_labels)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        logits = self.fc(self.dropout(hidden[-1]))
        return logits        