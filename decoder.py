import torch
import torch.nn as nn
from main import vocab, encode, decode, create_decoder_sequences, word2idx, idx2word
from attention import Attention, attention
from encoder import Encoder, outputs, hidden
class Decoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, attention = None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.attention = attention
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x, hidden, encoder_outputs = None):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded, hidden)
        if self.attention is not None and encoder_outputs is not None:
            context, attention_weights = self.attention(hidden, encoder_outputs)
            output = torch.cat([output, context], dim=2)
        prediction = self.fc(output)
        return prediction, hidden
    
decoder = Decoder(
    vocab_size=len(vocab),
    embedding_dim=8,
    hidden_dim=16,
    attention = attention
)

# word2idx["<SOS>"]

# decoder_input = torch.tensor([[word2idx["<SOS>"]]])

# prediction, hidden = decoder(
#     decoder_input,
#     hidden
# )
# predicted_word = prediction.argmax(dim=-1)
