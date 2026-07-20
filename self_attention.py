import torch
import torch.nn as nn
import torch.nn.functional as F
from main import vocab, encode, create_decoder_sequences, pairs, word2idx, idx2word, decode
import math
import numpy as np
vocab_size = len(vocab)
embedding_dim = 16
embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_dim
)


def translate(sentence, model, max_len=20):
    source = torch.tensor(
        encode(sentence),
        dtype=torch.long
    ).unsqueeze(0)
    encoder_output, _ = model.encoder(source)
    generated = [word2idx["<SOS>"]]
    for _ in range(max_len):
        decoder_input = torch.tensor(
            generated,
            dtype=torch.long
        ).unsqueeze(0)
        mask = model.make_target_mask(decoder_input)
        logits, _ = model.decoder(
            decoder_input,
            encoder_output,
            mask
        )
        next_logits = logits[:, -1, :]
        next_token = torch.argmax(next_logits, dim=-1).item()
        generated.append(next_token)
        if next_token == word2idx["<EOS>"]:
            break
    return decode(generated[1:])

def forecast(context_token, model, quantizer, prediction_length):
    source = torch.tensor(context_token, dtype=torch.long).unsqueeze(0)
    encoder_output, _ = model.encoder(source)
    generated = [256]
    for _ in range(prediction_length):
        decoder_input = torch.tensor(generated, dtype=torch.long).unsqueeze(0)
        mask = model.make_target_mask(decoder_input)
        logits, _ = model.decoder(decoder_input, encoder_output, mask) 
        next_logits = logits[:, -1, :]
        next_token = torch.argmax(next_logits, dim = -1).item()
        generated.append(next_token)
    predicted_token = np.array(generated[1:])
    forecast = quantizer.decode(predicted_token)
    return forecast


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        self.fc_out = nn.Linear(d_model, d_model)
    def forward(self, query, key, value, mask = None):
        batch_size = query.size(0)
        query_length = query.size(1)
        key_length = key.size(1)
        value_length = value.size(1)

        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)

        Q = Q.view(batch_size,
                query_length,
                self.num_heads,
                self.head_dim)

        K = K.view(batch_size,
                key_length,
                self.num_heads,
                self.head_dim)

        V = V.view(batch_size,
                value_length,
                self.num_heads,
                self.head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)


        scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2)
        context = context.reshape(batch_size, query_length, self.d_model)

        output = self.fc_out(context)
        return output, attention_weights

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
    def forward(self, x):
        return self.network(x)

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dff):
        super().__init__()
        self.self_attention = MultiheadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff=dff)
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x):
        attention_output, weights = self.self_attention(x, x, x)
        x = self.norm1(x + attention_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x, weights

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dff):
        super().__init__()
        self.masked_attention = MultiheadAttention(d_model, num_heads)
        self.cross_attention = MultiheadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff = dff)
    def forward(self, decoder_input, encoder_output, mask):
        attention_output, _ = self.masked_attention(decoder_input, decoder_input, decoder_input, mask)
        x = self.norm1(decoder_input + attention_output)
        cross_output, attention = self.cross_attention(x, encoder_output, encoder_output)
        x = self.norm2(x + cross_output)
        ff_output = self.ff(x)
        x = self.norm3(x + ff_output)

        return x, attention

        

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len = 100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.layers = nn.ModuleList([EncoderBlock(d_model=d_model, num_heads=num_heads, dff=d_ff) for _ in range(num_layers)])

    def forward(self, tokens):
        x = self.embedding(tokens)
        x = self.position(x)
        attention_maps = []
        for layer in self.layers:
            x, weights = layer(x)
            attention_maps.append(weights)
        return x, attention_maps


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len = 100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderBlock(d_model, num_heads, dff=d_ff) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
    def forward(self, target_tokens, encoder_outputs, mask):
        x = self.embedding(target_tokens)
        x = self.position(x)
        attention_maps = []
        for layer in self.layers:
            x, attention = layer(x, encoder_outputs, mask)
            attention_maps.append(attention)
        logits = self.fc_out(x)
        return logits, attention_maps

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len = 100):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_len = max_len)
        self.decoder = TransformerDecoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_len = max_len)
    def make_target_mask(self, target):
        seq_len = target.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=target.device))
        return mask.unsqueeze(0).unsqueeze(1)
    def forward(self, source, target):
        encoder_output, _ = self.encoder(source)
        decoder_output, _ = self.decoder(target, encoder_output, self.make_target_mask(target))
        return decoder_output


model = Transformer(vocab_size= vocab_size, d_model = 16, num_heads= 2, d_ff= 16, num_layers= 3)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

sentence = "I love dogs"
translation = "J aime les chiens"

input_embeddings = encode(sentence)
decode_sequence = create_decoder_sequences(translation)

source = torch.tensor(
    input_embeddings,
    dtype=torch.long
).unsqueeze(0)

decoder_input = torch.tensor(
    decode_sequence[0],
    dtype=torch.long
).unsqueeze(0)

decoder_target = torch.tensor(
    decode_sequence[1],
    dtype=torch.long
).unsqueeze(0)

output = model(
    source,
    decoder_input
)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

criterion = nn.CrossEntropyLoss()

training_data = []

for source_sentence, target_sentence in pairs:
    source = torch.tensor(
        encode(source_sentence),
        dtype=torch.long
    ).unsqueeze(0)
    decoder_input, decoder_target = create_decoder_sequences(
        target_sentence
    )
    decoder_input = torch.tensor(
        decoder_input,
        dtype=torch.long
    ).unsqueeze(0)
    decoder_target = torch.tensor(
        decoder_target,
        dtype=torch.long
    ).unsqueeze(0)
    training_data.append(
        (
            source,
            decoder_input,
            decoder_target
        )
    )

# for epoch in range(1):
#     total_loss = 0
#     for source, decoder_input, decoder_target in training_data:
#         output = model(source, decoder_input)
#         output = output.view(-1, len(vocab))
#         target = decoder_target.view(-1)
#         loss = criterion(output, target)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     if epoch % 50 == 0:
#         print(epoch, total_loss)

