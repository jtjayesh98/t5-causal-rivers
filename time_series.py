import pandas as pd
import numpy as np
# from self_attention import Transformer
from torch.utils.data import Dataset
import torch
data = pd.read_csv("./rivers_ts_east_germany.csv")


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

all_normalized = []

for column in data.columns[1:]:
    series = data[column].dropna().reset_index(drop = True)

    if len(series) == 0:
        continue

    mean = series.mean()
    std = series.std()

    normalized = (series - mean)/std

    split_idx = int(len(normalized) * 0.8)

    all_normalized.append(normalized[:split_idx])

all_normalized = np.concatenate(all_normalized)

quantizer = Quantizer(256)

quantizer.fit(all_normalized)



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

class MultiTimeSeriesDataset(Dataset):
    def __init__(self, dataframe, context_length, prediction_length, quantizer, sos_token, train = True, train_ratio = 0.8):
        super().__init__()
        self.samples = []
        for column in dataframe.columns[1:]:
            series = dataframe[column]
            series = series.dropna().reset_index(drop = True)
            if len(series) < context_length + prediction_length:
                continue
            mean = series.mean()
            std = series.std()
            if std == 0:
                continue
            series = (series - mean)/std
            tokens = quantizer.encode(series)
            split_idx = int(len(tokens) * train_ratio)
            if train:
                tokens = tokens[:split_idx]
            else:
                tokens = tokens[split_idx:]
            for i in range(len(tokens) - context_length - prediction_length + 1):
                encoder = tokens[i : i + context_length]
                future = tokens[i + context_length : i + context_length + prediction_length]
                decoder_input = [sos_token] + future[:-1].tolist()
                decoder_target = future
                self.samples.append((torch.tensor(encoder, dtype=torch.long), torch.tensor(decoder_input, dtype=torch.long), torch.tensor(decoder_target, dtype=torch.long)))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        return self.samples[index]
    
train_dataset = MultiTimeSeriesDataset(dataframe= data, context_length= 128, prediction_length= 32, quantizer= quantizer, sos_token= 256, train = True)

test_dataset = MultiTimeSeriesDataset(dataframe= data, context_length= 128, prediction_length= 32, quantizer= quantizer, sos_token= 256, train = False)


