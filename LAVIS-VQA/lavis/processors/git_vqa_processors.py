"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import re

from lavis.common.registry import registry
from lavis.processors.base_processor import BaseProcessor
from transformers import AutoProcessor
from omegaconf import OmegaConf


@registry.register_processor("git_vqa_image")
class GITVQAImageProcessor(BaseProcessor):
    def __init__(self, checkpoint):
        self.processor = AutoProcessor.from_pretrained(checkpoint)

    def __call__(self, item):
        return self.processor(images=item, return_tensors="pt").pixel_values

    @classmethod
    def from_config(cls, config=None):
        if config is None:
            config = OmegaConf.create()

        return cls(config["checkpoint"])
