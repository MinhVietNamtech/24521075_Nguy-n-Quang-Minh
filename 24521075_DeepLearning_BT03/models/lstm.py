from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence

class LSTM(nn.Module):
    def __init__(self, 
                vocab_size: int, 
                hidden_size: int, 
                n_layer: int, 
                n_labels: int,
                padding_idx: int = 0,
                dropout: float = 0.2
                ):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size, padding_idx=padding_idx)
        self.lstm = nn.LSTM(input_size=hidden_size, 
                            hidden_size=hidden_size, 
                            num_layers=n_layer, 
                            batch_first=True,
                            dropout=dropout if n_layer > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, n_labels)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        logits = self.fc(self.dropout(hidden[-1]))
        return logits
