import copy
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

IGNORE_INDEX = -100
PROMPT_DICT = {
    "prompt_input": "### Input:\n{input}\n\n### Instruction:\n{instruction}\n\n### Response:",
    "prompt_no_input": "### Instruction:\n{instruction}\n\n### Response:"
}


def preprocess_openllama(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    input_ids, labels = [], []
    for s in sources:
        s_tokens = tokenizer.encode(s)
        inpt = [1] + s_tokens + [2] + [1]
        inpt = inpt[-1024:]
        input_ids.append(torch.LongTensor(inpt))
    return input_ids


class SelfInstructTestDataset(Dataset):
    """Dataset for Generation on self-instruct test set"""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, **args):
        super(SelfInstructTestDataset, self).__init__()
        list_data_dict = jload(data_path)
        self.tokenizer = tokenizer
        self.tokenizer.bos_token_id = 1
        self.tokenizer.eos_token_id = 2
        self.tokenizer.pad_token_id = 2
        self.args = args

        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if 'input' in example else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]

        self.data = preprocess_openllama(sources, tokenizer)
        self.sources = sources
        self.answers = ['' for example in list_data_dict]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self.data[i], self.sources[i], self.answers[i]

    def collate(self, instances):
        assert len(instances) == 1
        instance, source, ans = instances[0]
        return {
            'input_ids': instance.unsqueeze(0),
            'prompt': source,
            'decoding_method': self.args['decoding_method'],
            'top_k': self.args['top_k'],
            'top_p': self.args['top_p'],
            'generate_len': self.args['generate_len'],
            'gpt4_answer': ans
        }
