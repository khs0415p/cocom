import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Union, Optional
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase
)
from transformers.modeling_outputs import CausalLMOutput

class CoComForPretrining(nn.Module):
    def __init__(
        self,
        decoder: PreTrainedModel,
        compressor: PreTrainedModel = None,
        compressor_tokenizer: PreTrainedTokenizerBase = None,
        decoder_tokenizer: PreTrainedTokenizerBase = None,
        is_light: bool = False,
        compression_length: int = None,
    ):
        super().__init__()
        self.compressor = compressor
        self.decoder = decoder
        self.compressor_tokenizer = compressor_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.is_light = is_light
        self.compression_length = compression_length

        self.compression_embed = nn.Embedding(compression_length, self.compressor.get_input_embeddings().embedding_dim)
        # Autocompressor는 초기값을 eos token으로 함.


    def get_input_embeddings(self):
        return self.compressor.get_input_embeddings(), self.decoder.get_input_embeddings()

    def resize_token_embeddings(self, compressor_tokenizer, decoder_tokenizer):
        if compressor_tokenizer == decoder_tokenizer:
            self.decoder.resize_token_embeddings(len(decoder_tokenizer))
        else:
            self.decoder.resize_token_embeddings(len(decoder_tokenizer))
            self.compressor.resize_token_embeddings(len(compressor_tokenizer))

    def forward(
        self,
        contexts: Union[torch.LongTensor, List[torch.LongTensor]],
        question: torch.LongTensor,
        labels: torch.LongTensor,
    ):
        min_length = min(self.compression_length, contexts.size(1))
        compression_token_ids = torch.arange(min_length, dtype=torch.long, device=contexts.device).unsqueeze(0).expand(contexts.size(0), -1)
        compression_embeds = self.compression_embed(compression_token_ids)
        compression_embeds = compression_embeds.to(self.compressor.dtype)

        compressor_input_ids = self.compressor.get_input_embeddings()(contexts)

        compression_input_embeds = torch.cat([compressor_input_ids, compression_embeds], dim=1)

        compression_output = self.compressor(inputs_embeds=compression_input_embeds, output_hidden_states=True)
        last_hidden_states = compression_output['hidden_states'][-1][:, -self.compression_length:]

        # TODO: dim issue (`is_light(bert)`: 768, `decoder`: 4096)
        length = labels.size(1)
        decoder_embeds = self.decoder.get_input_embeddings()(question)

        sep_embed = self.decoder.get_input_embeddings().weight[self.decoder_tokenizer.sep_token_id].unsqueeze(0).expand(decoder_embeds.size(0), -1).unsqueeze(1)
        last_hidden_states = torch.cat([last_hidden_states, sep_embed], dim=1)

        inputs_embeds = torch.cat([decoder_embeds, last_hidden_states], dim=1) # batch , (seq + com_len), dim
        decoder_output = self.decoder(inputs_embeds=inputs_embeds, output_hidden_states=True)

        last_hidden_states = decoder_output['hidden_states'][-1][:, -length:, :]
        logits = self.decoder.lm_head(last_hidden_states)


        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, len(self.decoder_tokenizer)), shift_labels.view(-1))
        
        output = CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=decoder_output['hidden_states'],
        )

        return output

    def print_trainable_parameters(self):
        print(f"=== Decoder trainable parameters ===")
        self.decoder.print_trainable_parameters()
        print(f"=== Compressor trainable parameters ===")
        if id(self.decoder) == id(self.compressor):
            print("TheCompressor is the same as the Decoder")
        else:
            total_params = sum(p.numel() for p in self.compressor.parameters())
            trainable_params = sum(p.numel() for p in self.compressor.parameters() if p.requires_grad)
            print(f"trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {trainable_params/total_params}")

    def save(self, output_path):
        compressor_path = os.path.join(output_path, 'compressor')
        decoder_path = os.path.join(output_path, 'decoder')

        os.makedirs(decoder_path, exist_ok=True)

        self.decoder.save_pretrained(decoder_path)
        self.decoder_tokenizer.save_pretrained(decoder_path)

        if self.is_light:
            os.makedirs(compressor_path, exist_ok=True)
            self.compressor.save_pretrained(compressor_path)
            self.compressor_tokenizer.save_pretrained(compressor_path)