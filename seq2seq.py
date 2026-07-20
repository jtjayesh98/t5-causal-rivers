import torch
import torch.nn as nn

from decoder import decoder
from encoder import encoder
from main import vocab, create_decoder_sequences, encode, decode, idx2word, word2idx, pairs


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, decoder_input):

        batch_size = source.shape[0]
        target_length = decoder_input.shape[1]
        vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size,
                            target_length,
                            vocab_size)

        encoder_outputs, hidden = self.encoder(source)

        current_input = decoder_input[:, 0].unsqueeze(1)

        for t in range(target_length):

            prediction, hidden = self.decoder(
                current_input,
                hidden,
                encoder_outputs
            )

            outputs[:, t] = prediction.squeeze(1)

            if t + 1 < target_length:
                current_input = decoder_input[:, t+1].unsqueeze(1)

        return outputs
    
source = torch.tensor([encode("i love dogs")])

decoder_input, decoder_target = create_decoder_sequences(
    "j aime les chiens"
)

decoder_input = torch.tensor([decoder_input])
decoder_target = torch.tensor([decoder_target])


model = Seq2Seq(encoder, decoder)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)



for epoch in range(500):

    total_loss = 0

    for source_text, target_text in pairs:

        source = torch.tensor([encode(source_text)])

        decoder_input, decoder_target = create_decoder_sequences(target_text)

        decoder_input = torch.tensor([decoder_input])
        decoder_target = torch.tensor([decoder_target])

        outputs = model(
            source,
            decoder_input
        )

        outputs = outputs.reshape(
            -1,
            outputs.shape[-1]
        )

        decoder_target = decoder_target.reshape(-1)

        loss = criterion(
            outputs,
            decoder_target
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    if epoch % 50 == 0:
        print(
            epoch,
            total_loss
        )


def translate(sentence, model, max_len=10):

    model.eval()

    with torch.no_grad():

        source = torch.tensor([encode(sentence)])

        encoder_outputs, hidden = model.encoder(source)

        decoder_input = torch.tensor([[word2idx["<SOS>"]]])

        generated = []

        for _ in range(max_len):

            prediction, hidden = model.decoder(
                decoder_input,
                hidden,
                encoder_outputs
            )

            next_token = prediction.argmax(-1)

            token = next_token.item()

            if token == word2idx["<EOS>"]:
                break

            generated.append(idx2word[token])

            decoder_input = next_token

    return " ".join(generated)

print(translate("i love dogs", model))