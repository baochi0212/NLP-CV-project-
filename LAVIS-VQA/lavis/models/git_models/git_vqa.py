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
class GITVQA(BaseModel, AutoModelForCausalLM):
    """
    TextVQA model
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "base_textvqa": "configs/models/git/git_base_textvqa.yaml",
        "large_textvqa": "configs/models/git/git_large_textvqa.yaml",
        "base_vqa2": "configs/models/git/git_base_vqa2.yaml",
        "large_vqa2": "configs/models/git/git_large_vqa2.yaml",
    }

    def __init__(self, checkpoint, answer_space_size):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint)
        self.fc = nn.Linear(768, answer_space_size)
        self.loss_fn = nn.CrossEntropyLoss()

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
        # TODO: finish this function
        inputs = self.processor(samples["text_input"], images=samples["image"], padding="max_length", max_length=512, return_tensors="pt")
        targets = samples["answer"]
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state[:, -1, :]
        logits = self.fc(last_hidden_state)
        loss = self.loss_fn(logits, targets)

        return loss

    def predict_answers(
        self,
        samples,
        inference_method="generate",
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

        if inference_method == "generate":
            questions = samples["text_input"]

            batch_input_ids = self.processor(text=questions, add_special_tokens=False).input_ids
            max_len = max(len(input_ids) for input_ids in batch_input_ids)
            batch_input_ids = torch.tensor(
                [[self.processor.tokenizer.pad_token_id]*(max_len - len(input_ids)) + [self.processor.tokenizer.cls_token_id] + input_ids
                for input_ids in batch_input_ids]
            )

            batch_generated_ids = self.model.generate(pixel_values=samples["image"], input_ids=batch_input_ids, max_length=50, min_length=min_len)
            answer_ids = [ batch_generated_ids[i][len(batch_input_ids[i]):] for i in range(len(batch_input_ids)) ]
            pred_answers = self.processor.batch_decode(answer_ids, skip_special_tokens=True)

        return pred_answers

    @classmethod
    def from_config(cls, config):
        model = cls(config["checkpoint"], config["answer_space_size"])

        return model