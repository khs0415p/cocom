import torch

from typing import Any
from torch.utils.data import Dataset


class PretrainingDataset(Dataset):
    def __init__(self, compression_dataset, decoder_dataset):
        super().__init__()
        self.compression_data = compression_dataset
        self.decoder_data = decoder_dataset
        assert len(compression_dataset) == len(decoder_dataset)
        self.length = len(self.compression_data)

    def __getitem__(self, index) -> Any:
        
        return {
            "compression_inputs": self.compression_data[index],
            "decoder_inputs": self.decoder_data[index],
        }
    
    def __len__(self):
        return self.length