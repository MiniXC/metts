from torch.utils.data import Dataset
import torch
import numpy as np
import lco

class DummyDataset(Dataset):
    def __init__(self):
        self.phones = torch.randint(0, 100, (8*100, lco["max_lengths"]["phone"]))
        self.durations = torch.randint(0, 10, (8*100, lco["max_lengths"]["phone"])).numpy()
        self.audio = torch.randn((8*100, lco["max_lengths"]["audio"]))

    def __len__(self):
        return len(self.phones) * 10

    def __getitem__(self, idx):
        return {
            "phones": self.phones[idx % 10],
            "durations": self.durations[idx % 10],
            "audio": self.audio[idx % 10],
        }