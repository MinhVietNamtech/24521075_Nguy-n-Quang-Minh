from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence

# class LSTM_block():
#     def __init__(self, input_size, hidden_size, bias=True):
#         super(LSTM_block, self).__init__()
#         self.sigmoid = nn.Sigmoid()
#         self.tanh = nn.Tanh()

#         # W_x
#         self.W_xi = torch.rand((input_size, hidden_size))
#         self.W_xf = torch.rand((input_size, hidden_size))
#         self.W_xo = torch.rand((input_size, hidden_size))
#         self.W_xc = torch.rand((input_size, hidden_size))

#         # W_h
#         self.W_hi = torch.rand((hidden_size, hidden_size))
#         self.W_hf = torch.rand((hidden_size, hidden_size))
#         self.W_ho = torch.rand((hidden_size, hidden_size))
#         self.W_hc = torch.rand((hidden_size, hidden_size))

#         # biases
#         self.b_i = torch.rand((1,hidden_size))
#         self.b_f = torch.rand((1,hidden_size))
#         self.b_o = torch.rand((1,hidden_size))
#         self.b_c = torch.rand((1,hidden_size))

#     def forward(self, X): # X: (len, input_size)
#         # Initialize hidden and cell states
#         hidden_states = []
#         cell_states = []
#         batch_size = X.size(1)
#         hidden_state = torch.zeros(batch_size, self.W_hi.size(0))
#         prev_cell = torch.zeros(batch_size, self.W_hi.size(0))
#         for word in X:
#             I_t = self.sigmoid(torch.matmul(word,self.W_xi) + torch.matmul(hidden_state,self.W_hi) + self.b_i)
#             F_t = self.sigmoid(torch.matmul(word,self.W_xf) + torch.matmul(hidden_state,self.W_hf) + self.b_f)
#             O_t = self.sigmoid(torch.matmul(word,self.W_xo) + torch.matmul(hidden_state,self.W_ho) + self.b_o)

#             temp_cell = self.tanh(torch.matmul(word,self.W_xc) + torch.matmul(hidden_state, self.W_hc) + self.b_c)
#             cur_cell = F_t * prev_cell + I_t * temp_cell
#             hidden_state = O_t * self.tanh(cur_cell)
#             hidden_states.append(hidden_state)
#             cell_states.append(cur_cell)
#             prev_cell = cur_cell # C_t-1 = C_t
#         return O_t, (hidden_states, cell_states)

#     def backward(self, d_O_t, )

# class LSTM5(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers=5):
#         super(LSTM5, self).__init__()
#         self.layers = nn.Sequential(
#             nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,)
#             for _ in range(num_layers)
#         )

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
