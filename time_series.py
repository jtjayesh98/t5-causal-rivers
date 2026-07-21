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

for column in data.columns[1:5]:
    print(column)
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
    def __init__(
        self,
        dataframe,
        context_length,
        prediction_length,
        quantizer,
        sos_token,
        train=True,
        train_ratio=0.8,
    ):
        super().__init__()

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.sos_token = sos_token

        # Store one token array per time series
        self.series_tokens = []

        # Maps dataset index -> (series_index, window_start)
        self.index = []

        for column in dataframe.columns[1:]:
            series = dataframe[column].dropna().reset_index(drop=True)

            if len(series) < context_length + prediction_length:
                continue

            mean = series.mean()
            std = series.std()

            if std == 0:
                continue

            series = (series - mean) / std
            tokens = quantizer.encode(series).astype(np.int64)

            split_idx = int(len(tokens) * train_ratio)

            if train:
                tokens = tokens[:split_idx]
            else:
                tokens = tokens[split_idx:]

            if len(tokens) < context_length + prediction_length:
                continue

            series_id = len(self.series_tokens)
            self.series_tokens.append(tokens)

            n_windows = len(tokens) - context_length - prediction_length + 1

            for start in range(n_windows):
                self.index.append((series_id, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        series_id, start = self.index[idx]

        tokens = self.series_tokens[series_id]

        encoder = tokens[start:start + self.context_length]

        future = tokens[
            start + self.context_length:
            start + self.context_length + self.prediction_length
        ]

        decoder_input = np.empty(self.prediction_length, dtype=np.int64)
        decoder_input[0] = self.sos_token
        decoder_input[1:] = future[:-1]

        return (
            torch.from_numpy(encoder),
            torch.from_numpy(decoder_input),
            torch.from_numpy(future),
        )
    
train_dataset = MultiTimeSeriesDataset(dataframe= data, context_length= 128, prediction_length= 32, quantizer= quantizer, sos_token= 256, train = True)

test_dataset = MultiTimeSeriesDataset(dataframe= data, context_length= 128, prediction_length= 32, quantizer= quantizer, sos_token= 256, train = False)

from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)