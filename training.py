from self_attention import Transformer, forecast
from time_series import train_loader, test_loader
import torch.nn as nn
import torch

device = ("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(
    vocab_size=257,
    d_model=64,
    num_heads=4,
    d_ff=256,
    num_layers=4,
    max_len=512
)
model.to(device)



criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

print("Starting Training")

for epoch in range(10):
    total_loss = 0
    for encoder_input, decoder_input, decoder_target in train_loader:
        encoder_input = encoder_input.to(device, non_blocking=True)
        decoder_input = decoder_input.to(device, non_blocking=True)
        decoder_target = decoder_target.to(device, non_blocking=True)

        output = model(
            encoder_input,
            decoder_input
        )
        output = output.reshape(-1, 257)
        decoder_target = decoder_target.reshape(-1)
        loss = criterion(
            output,
            decoder_target
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(epoch, total_loss / len(train_loader))


model.eval()

test_loss = 0

with torch.no_grad():
    for encoder_input, decoder_input, decoder_target in test_loader:

        encoder_input = encoder_input.to(device, non_blocking=True)
        decoder_input = decoder_input.to(device, non_blocking=True)
        decoder_target = decoder_target.to(device, non_blocking=True)
        output = model(encoder_input, decoder_input)

        output = output.reshape(-1, 257)
        decoder_target = decoder_target.reshape(-1)

        loss = criterion(
            output,
            decoder_target
        )
        test_loss += loss.item()

    print(f"Test Loss: {test_loss / len(test_loader):.4f}")