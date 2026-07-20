import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, decoder_hidden, encoder_outputs):
        scores = torch.bmm(
            encoder_outputs,
            decoder_hidden.transpose(1,2)
        )
        attention_weights = F.softmax(
            scores,
            dim=1
        )
        context = torch.bmm(
            attention_weights.transpose(1,2),
            encoder_outputs
        )
        return context,attention_weights
    

attention = Attention()