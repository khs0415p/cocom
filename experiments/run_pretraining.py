import os
import sys
import torch
import logging
import datasets

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils import (
    CocomTrainer,
    CustomDataCollator
)
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    HfArgumentParser,
    set_seed,
)
from models import LlamaForCausalLM, CoComForPretrining
from data import load_pretrain_raw_dataset, preprocess_pretrain_datasets
from dataset import PretrainingDataset
from arguments import ModelArguments, DataArguments, HfTrainingArguments


"""
Pre - training
learning Rate 1e-4
lr scheduler type linear
warmup ratio 0.05
weight dacay 0.1
overall batch size 256
optimizer AdamW
epochs 1
LoRa layers all linear layers
LoRa alpha 32
LoRa dropout 0.1
LoRa 𝑟 16
LoRa bias None
GPU 8 x A100 80GB
context max length 128
"""


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, HfTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    if model_args.compressor_name_or_path is None:
        model_args.compressor_name_or_path = model_args.decoder_name_or_path

    set_seed(training_args.seed)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "trust_remote_code": model_args.trust_remote_code,
        "use_fast": model_args.use_fast,
    }

    decoder_tokenizer = AutoTokenizer.from_pretrained(
                    model_args.decoder_name_or_path,
                    **config_kwargs,
                )
    if model_args.compressor_name_or_path == model_args.decoder_name_or_path:
        special_tokens_dict = {
            "sep_token": "[SEP]",
            "additional_special_tokens": ["<AE>", "<CTX>"],
        }

        decoder_tokenizer.add_special_tokens(
            special_tokens_dict=special_tokens_dict
        )

        compressor_tokenizer = decoder_tokenizer
    else:
        special_tokens_dict = {
            "sep_token": "[SEP]",
        }

        decoder_tokenizer.add_special_tokens(
            special_tokens_dict=special_tokens_dict
        )

        compressor_tokenizer =  AutoTokenizer.from_pretrained(
            model_args.compressor_name_or_path,
            **config_kwargs,
        )

        special_tokens_dict = {
            "additional_special_tokens": ["<AE>", "<CTX>"],
        }

        compressor_tokenizer.add_special_tokens(
            special_tokens_dict=special_tokens_dict
        )

    if decoder_tokenizer.pad_token is None:
        decoder_tokenizer.pad_token = decoder_tokenizer.eos_token

    ### Model
    config_kwargs.pop("use_fast")
    config_kwargs.update({"torch_dtype": torch.bfloat16})
    config_kwargs.update({"attn_implementation": model_args.attn_implementation})
    decoder = LlamaForCausalLM.from_pretrained(model_args.decoder_name_or_path, **config_kwargs)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['o_proj', 'q_proj', 'up_proj', 'down_proj', 'gate_proj', 'k_proj', 'v_proj'],
    )

    peft_model = get_peft_model(decoder, lora_config)
    peft_model.use_cache = False
    decoder = peft_model

    if ('llama' in model_args.compressor_name_or_path) and model_args.is_light:
        raise ValueError(
            "if `is_light` is True, the compressor's type must be only-encoder. "
            f"But, now `is_light` is {model_args.is_light} and decoder's name is {model_args.compressor_name_or_path}."
            )

    if model_args.is_light:
        config_kwargs.pop("torch_dtype")
        compressor = AutoModel.from_pretrained(model_args.compressor_name_or_path, **config_kwargs)
    else:
        compressor = decoder
    
    compression_length = data_args.context_length // model_args.compression_rate
    model = CoComForPretrining(
        compressor=compressor,
        decoder=decoder,
        compressor_tokenizer=compressor_tokenizer,
        decoder_tokenizer=decoder_tokenizer,
        is_light=model_args.is_light,
        compression_length=compression_length,
    )

    model.print_trainable_parameters()

    model.resize_token_embeddings(compressor_tokenizer, decoder_tokenizer)

    ### Data
    raw_datasets = load_pretrain_raw_dataset(data_args, model_args)

    compression_lm_datasets = preprocess_pretrain_datasets(raw_datasets, compressor_tokenizer, data_args, model_args)
    compression_train_dataset = compression_lm_datasets["train"]
    max_train_samples = min(len(compression_train_dataset), data_args.max_train_samples)
    compression_train_dataset = compression_train_dataset.shuffle(seed=training_args.seed).select(range(max_train_samples))

    compression_eval_dataset = compression_lm_datasets["validation"]
    max_eval_samples = min(len(compression_eval_dataset), data_args.max_eval_samples)
    compression_eval_dataset = compression_eval_dataset.select(range(max_eval_samples))

    if model_args.is_light:
        decoder_lm_datasets = preprocess_pretrain_datasets(raw_datasets, decoder_tokenizer, data_args, model_args)

        decoder_train_dataset = decoder_lm_datasets["train"]
        max_train_samples = min(len(decoder_train_dataset), data_args.max_train_samples)
        decoder_train_dataset = decoder_train_dataset.shuffle(seed=training_args.seed).select(range(max_train_samples))

        decoder_eval_dataset = decoder_lm_datasets["validation"]
        max_eval_samples = min(len(decoder_eval_dataset), data_args.max_eval_samples)
        decoder_eval_dataset = decoder_eval_dataset.select(range(max_eval_samples))

    else:
        decoder_train_dataset = compression_train_dataset
        decoder_eval_dataset = compression_eval_dataset

    train_dataset = PretrainingDataset(compression_train_dataset, decoder_train_dataset)
    eval_dataset = PretrainingDataset(compression_eval_dataset, decoder_eval_dataset)

    data_collator = CustomDataCollator(compressor_tokenizer.pad_token_id, decoder_tokenizer.pad_token_id)

    trainer = CocomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()

if __name__ == "__main__":
    train()