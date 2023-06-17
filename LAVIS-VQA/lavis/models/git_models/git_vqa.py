"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
import torch.nn as nn
from itertools import chain
from lavis.common.registry import registry
from lavis.models.base_model import BaseModel
from transformers import AutoProcessor, AutoModelForCausalLM


@registry.register_model("git_vqa")
class GITVQA(BaseModel):
    """
    TextVQA model
    """

    PRETRAINED_MODEL_CONFIG_DICT = {"base": "configs/models/pnp-vqa/pnp_vqa_base.yaml"}

    def __init__(self, config):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(config["checkpoint"])
        self.model = AutoModelForCausalLM.from_pretrained(config["checkpoint"])

    def forward(self, samples):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W). Default H=480, W=480.
                - text_input (list): A list of strings, each string is a question
                - answer (list): A list of strings, each string is an answer
                - weight (torch.Tensor): A tensor used to weigh each answer in the loss computation.
                   The shape of the tensor is (sum(n_answers),)
                - n_answers (torch.Tensor): A tensor shape (batch_size,) containing the number of answers
                     for each question in the batch.

        Returns:
            A GITOutput object containing loss and intermediate outputs,
            see :class:`lavis.models.git_outputs.GITOutput` for more details.
        """

        

        return

    def predict_answers(
        self,
        samples,
        max_len=20,
        min_len=0,
        **kwargs
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W). Default H=480, W=480.
                - text_input (str or [str]): String or a list of strings, each string is a question.
                                             The number of questions must be equal to the batch size. If a single string, will be converted to a list of string, with length 1 first.
        Returns:
            List: A list of strings, each string is an answer.
        """

        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]

        assert len(samples["text_input"]) == samples["image"].size(
            0
        ), "The number of questions must be equal to the batch size."

        torch.cuda.empty_cache()

        pixel_values = self.processor(images=samples["image"], return_tensors="pt").pixel_values

        question = samples["input_text"]
        input_ids = self.processor(text=question, add_special_tokens=False).input_ids
        input_ids = [self.processor.tokenizer.cls_token_id] + input_ids
        input_ids = torch.tensor(input_ids).unsqueeze(0)

        generated_ids = self.model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=max_len, min_length=min_len)

        pred_answers = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return pred_answers

    @classmethod
    def from_config(cls, model_config):
        model = cls(model_config)

        return model