from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self,
                vocab_size: int, 
                hidden_size: int, 
                n_layer: int, 
                n_labels: int,
                padding_idx: int = 0,
                dropout: float = 0.2):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)
        self.biLSTM = nn.LSTM(input_size=hidden_size,
                              hidden_size=hidden_size,
                              num_layers=n_layer,
                              batch_first=True,
                              dropout=dropout if n_layer>1 else 0.0,
                              bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2*hidden_size, n_labels)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.biLSTM(packed)
        output, _ = pad_packed_sequence(output, batch_first=True,
                                        total_length=x.size(1))
        return self.fc(self.dropout(output))
        

