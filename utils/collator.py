import torch
from dataclasses import dataclass

@dataclass
class CustomDataCollator:
    def __init__(self, compression_pad_token_id, decoder_pad_token_id):
        self.colums = ['compression_inputs', 'decoder_inputs', 'lm_compression_inputs', 'lm_decoder_inputs', 'lm_decoder_labels']
        self.compression_pad_token_id = compression_pad_token_id
        self.decoder_pad_token_id = decoder_pad_token_id

    def __call__(self, batch):
        compression_input_ids = torch.stack([torch.LongTensor(instance[self.colums[0]]["input_ids"]) for instance in batch], dim=0)
        decoder_input_ids = torch.stack([torch.LongTensor(instance[self.colums[1]]["input_ids"]) for instance in batch], dim=0)
        decoder_labels = torch.stack([torch.LongTensor(instance[self.colums[1]]["labels"]) for instance in batch], dim=0)

        lm_compression_input_dis = torch.stack([torch.LongTensor(instance[self.colums[2]]["input_ids"]) for instance in batch], dim=0)
        lm_decoder_input_ids = torch.stack([torch.LongTensor(instance[self.colums[3]]["input_ids"]) for instance in batch], dim=0)
        lm_decoder_labels = torch.stack([torch.LongTensor(instance[self.colums[4]]["labels"]) for instance in batch], dim=0)

        return {
            "ae_compression_input_ids": compression_input_ids,
            "ae_decoder_input_ids": decoder_input_ids,
            "ae_decoder_labels": decoder_labels,
            "lm_compression_input_ids": lm_compression_input_dis,
            "lm_decoder_input_ids": lm_decoder_input_ids,
            "lm_decoder_labels": lm_decoder_labels,
        }