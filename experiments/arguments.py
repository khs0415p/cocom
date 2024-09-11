import torch

from typing import Optional
from dataclasses import dataclass, field
from transformers import TrainingArguments as HfTrainingArguments

@dataclass
class ModelArguments:
    compressor_name_or_path: str = field(
        default=None,
        metadata={
            "help": (
                "The compressor model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )

    decoder_name_or_path: str = field(
        default=None,
        metadata={
            "help": (
                "The decoder model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        }
    )

    compression_rate: int = field(
        default=4,
        metadata={
            "help": (
                "The compression rate for context embedding. "
                "Example) context embeddings = (context length // compression rate)."
            )
        }
    )

    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Where do you want to store the pretrained models downloaded from huggingface.co"
            )
        }
    )

    attn_implementation: Optional[str] = field(
        default="flash_attention_2",
        metadata={
            "choices": ["eager", "sdpa", "flash_attention_2"],
            "help": (
                "The method for attention calculation. Default is `flash_attention_2`. if you want to change the method. select among `eagar`, `sdpa`, `flash_attention_2`"
            )
        }
    )

    use_fast: Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                "Whether to use a fast Rust-based tokenizer if supported by a given model. "
                "If a quick tokenizer is not available for a given model, a regular Python-based tokenizer is returned instead."
                )
        }
    )

    trust_remote_code: Optional[bool] = field(
        default=False,
    )

    peft_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to the lora model."}
    )

    is_light: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If this option is true, Use BERT as compressor model."
        }
    )


@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )

    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )

    context_length: Optional[int] = field(
        default=128,
        metadata={
            "help": ("Maximum number of tokens to be provided to the model. ")
        },
    )

    num_workers: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of workers for parallel processing of `datasets`"
        }
    )