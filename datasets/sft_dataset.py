#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import os
import json
from tqdm import tqdm
import ipdb
import random
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass, field
from typing import Callable, Dict, Sequence

import torch
import torch.distributed as dist
import transformers
from torch.utils.data import Dataset
from tqdm import tqdm

PROMPT_DICT = {
    "prompt_input": "### Input:\n{input}\n\n### Instruction:\n{instruction}\n\n### Response:",
    "prompt_no_input": "### Instruction:\n{instruction}\n\n### Response:"
}

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    max_length: str
) -> Dict:
    """Preprocess the data by tokenizing."""
    input_ids, labels = [], []
    for s, t in zip(sources, targets):
        s_tokens = tokenizer.encode(s)
        t_tokens = tokenizer.encode(t)
        '''
            bos_token_id is 1
            eos_token_id is 2
        '''
        inpt = [1] + s_tokens + [2] + [1] + t_tokens + [2]
        label = [1] + [-100] * len(s_tokens) + [2] + [1] + t_tokens + [2]
        inpt = inpt[-max_length:]
        label = label[-max_length:]
        input_ids.append(torch.LongTensor(inpt))
        labels.append(torch.LongTensor(label))
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, max_length: int):
        super(SupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path))
        self.tokenizer = tokenizer
        # for openllama tokenizer
        self.tokenizer.bos_token_id = 1    #  eos_token_id is 2
        self.tokenizer.eos_token_id = 2    #  eos_token_id is 2

        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in tqdm(list_data_dict)
        ]
        targets = [example['output'] for example in list_data_dict]

        data_dict = preprocess(sources, targets, tokenizer, max_length)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        print(f'[!] collect {len(self.input_ids)} samples for training')

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def collate(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.eos_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.eos_token_id),
        )