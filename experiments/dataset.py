import torch

from typing import Any
from torch.utils.data import Dataset
from utils import CTX_NUM_MAPPING

class PretrainingDataset(Dataset):
    def __init__(
        self,
        compression_dataset,
        decoder_dataset,
        tokenizer,
        compression_rate,
        context_length
    ):
        super().__init__()
        if compression_rate not in CTX_NUM_MAPPING:
            raise ValueError("Choose the `compression_rate` between 4, 16 and 128.")
        
        ctx_num = CTX_NUM_MAPPING[compression_rate]
        num_extra_token = ctx_num + 3
        max_length = context_length - num_extra_token
        ae_token_id = tokenizer.encode("<AE>", add_special_tokens=False)
        ctx_token_id = tokenizer.encode("<CTX>", add_special_tokens=False)

        self.prepend_token_ids = [tokenizer.bos_token_id] + ae_token_id
        self.append_token_ids = (ctx_token_id * ctx_num) + [tokenizer.eos_token_id]

        self.compression_data = compression_dataset
        self.decoder_data = decoder_dataset
        assert len(compression_dataset) == len(decoder_dataset)
        self.length = len(self.compression_data)

    def __getitem__(self, index) -> Any:
        compression_inputs = {"input_ids": self.prepend_token_ids + self.compression_data[index]["input_ids"] + self.append_token_ids}
        decoder_inputs = {
            "input_ids": self.prepend_token_ids + self.decoder_data[index]["input_ids"] + self.append_token_ids,
            "labels": self.prepend_token_ids + self.decoder_data[index]["input_ids"].copy() + self.append_token_ids,
            }

        half = len(self.compression_data[index]['input_ids']) // 2

        lm_compression_inputs = {'input_ids': self.compression_data[index]['input_ids'][:half]}
        lm_decoder_labels = {'labels': self.decoder_data[index]['labels'][half:]}

        return {
            "compression_inputs": compression_inputs,
            "decoder_inputs": decoder_inputs,
            "lm_compression_inputs": lm_compression_inputs,
            "lm_decoder_inputs": self.decoder_data[index],
            "lm_decoder_labels": lm_decoder_labels,
        }
    
    def __len__(self):
        return self.length