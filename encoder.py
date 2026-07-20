import torch
import torch.nn as nn
from main import vocab, encode, decode, create_decoder_sequences

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden
    
embedding_dim = 8
hidden_dim = 16

encoder = Encoder(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim
)

source = encode("i love dogs")

source = torch.tensor(source)
source = source.unsqueeze(0)
outputs, hidden = encoder(source)
