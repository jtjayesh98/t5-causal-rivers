import pandas as pd
import numpy as np
# from self_attention import Transformer
from torch.utils.data import Dataset
import torch
data = pd.read_csv("./rivers_ts_east_germany.csv")

series = data[data.columns[1]]

series = series.dropna().reset_index(drop=True)

mean = series.mean()
std = series.std()

normalized = (series - mean) / std

class Quantizer:

    def __init__(self, num_bins):
        self.num_bins = num_bins
    def fit(self, series):

        self.minimum = series.min()
        self.maximum = series.max()

        self.edges = np.linspace(
            self.minimum,
            self.maximum,
            self.num_bins + 1
        )

    def encode(self, series):
        tokens = np.digitize(series, self.edges) - 1
        tokens = np.clip(tokens, 0, self.num_bins - 1)
        return tokens
    
    def decode(self, tokens):
        left = self.edges[tokens]
        right = self.edges[tokens + 1]

        value = (left + right) / 2
        return value
    
quantizer = Quantizer(num_bins=256)

quantizer.fit(series)

tokens = quantizer.encode(series)

split_idx = int(len(tokens) * 0.8)

train_tokens = tokens[:split_idx]
test_tokens = tokens[split_idx:]

reconstructed = quantizer.decode(tokens)

class TimeSeriesDataset(Dataset):
    def __init__(self, tokens, context_length, prediction_length, sos_tokens):
        super().__init__()
        self.tokens = tokens
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.sos_tokens = sos_tokens
    def __len__(self):
        return (len(self.tokens) - self.context_length - self.prediction_length + 1)
    
    def __getitem__(self, index):
        encoder = self.tokens[index : index + self.context_length]
        future = self.tokens[index + self.context_length : index + self.context_length + self.prediction_length]
        decoder_input = [self.sos_tokens] + future[:-1].tolist()
        decoder_target = future

        return (torch.tensor(encoder, dtype=torch.long), torch.tensor(decoder_input, dtype=torch.long), torch.tensor(decoder_target, dtype=torch.long),)
     

dataset = TimeSeriesDataset(tokens, 128, 32, 256)

train_dataset = TimeSeriesDataset(train_tokens, 128, 32, 256)
test_dataset = TimeSeriesDataset(test_tokens, 128, 32, 256)

from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)