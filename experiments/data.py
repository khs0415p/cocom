import datasets
import logging

from itertools import chain
from utils import CTX_NUM_MAPPING

logger = logging.getLogger(__name__)

def load_pretrain_raw_dataset(data_args, model_args):
    raw_datasets = datasets.load_dataset(
        data_args.dataset_name,
        cache_dir=model_args.cache_dir
        )
    
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = datasets.load_dataset(
            data_args.dataset_name,
            # split=f"train[:{data_args.validation_split_percentage}%]",
            split=f"train[:1%]",
            cache_dir=model_args.cache_dir,
        )
        raw_datasets["train"] = datasets.load_dataset(
            data_args.dataset_name,
            # split=f"train[{data_args.validation_split_percentage}%:]",
            split=f"train[1%:2%]",
            cache_dir=model_args.cache_dir,
        )
    return raw_datasets

def preprocess_pretrain_datasets(raw_datasets, tokenizer, data_args, model_args):
    # Using dmrau/kilt-128 datasets
    column_names = raw_datasets['train'].column_names
    text_columns = ["content", "next_text"]
    ae_token_id = tokenizer.encode("<AE>", add_special_tokens=False)
    ctx_token_id = tokenizer.encode("<CTX>", add_special_tokens=False)

    def tokenize_function(examples):
        texts = [ctx + n_ctx for ctx, n_ctx in zip(examples[text_columns[0]], examples[text_columns[1]])]
        output = tokenizer(texts, add_special_tokens=False)
        return output
    
    if model_args.compression_rate not in CTX_NUM_MAPPING:
        raise ValueError("Choose the `compression_rate` between 4, 16 and 128.")

    ctx_num=CTX_NUM_MAPPING[model_args.compression_rate]
    num_extra_token = ctx_num + 3
    max_length = data_args.context_length - num_extra_token

    tokenized_dataset = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.num_workers,
        remove_columns=column_names,
        desc="Tokenizing..."
        )
    
    def group_texts(examples, chunk_size):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        if total_length >= chunk_size:
            total_length = (total_length // chunk_size) * chunk_size

        prepend_token_ids = [tokenizer.bos_token_id] + ae_token_id
        append_token_ids = (ctx_token_id * ctx_num) + [tokenizer.eos_token_id]
        result = {
            k: [prepend_token_ids + t[i : i + chunk_size] + append_token_ids for i in range(0, total_length, chunk_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_dataset.map(
        lambda x:group_texts(x, chunk_size=max_length),
        batched=True,
        num_proc=data_args.num_workers,
        desc="Grouping..."
        )
    
    return lm_datasets