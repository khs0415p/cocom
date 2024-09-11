import os
import random
import torch
import torch.nn as nn

from typing import Dict, Any, Union, Optional
from transformers import Trainer
from models import CoComForPretrining

class CocomTrainer(Trainer):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.model.save(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def compute_loss(
        self,
        model: CoComForPretrining,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_output: bool = False,
    ):
        if random.random() > 0.5:
            output = model(inputs['ae_compression_input_ids'], inputs['ae_decoder_input_ids'], inputs['ae_decoder_labels'])
        else:
            output = model(inputs['lm_compression_input_ids'], inputs['lm_decoder_input_ids'], inputs['lm_decoder_labels'])

        return (output.loss, output) if return_output else output.loss