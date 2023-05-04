from header import *
from .samplers import DistributedBatchSampler
from .sft_dataset import *
from .self_instruct_test_dataset import *

def get_tokenizer(model):
    tokenizer = LlamaTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_sft_dataset(args):
    tokenizer = get_tokenizer(args['tokenizer'])
    dataset_name = args['models'][args['model']]['stage1_train_dataset']
    data_path = args["data_path"]
    additional_params = {}
    data = globals()[dataset_name](data_path, tokenizer, **additional_params)

    sampler = torch.utils.data.RandomSampler(data)
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    batch_size = args['world_size'] * args['dschf'].config['train_micro_batch_size_per_gpu']
    batch_sampler = DistributedBatchSampler(
        sampler, 
        batch_size,
        True,
        rank,
        world_size
    )
    iter_ = DataLoader(
        data, 
        batch_sampler=batch_sampler, 
        num_workers=1,
        collate_fn=data.collate, 
        pin_memory=True
    )
    return data, iter_, sampler

def load_sft_test_dataset(args):
    tokenizer = get_tokenizer(args['tokenizer'])
    dataset_name = args['models'][args['model']]['test_dataset']
    data_path = args["data_path"]
    additional_params = {}
    additional_params['decoding_method'] = args['decoding_method']
    additional_params['top_k'] = args['top_k']
    additional_params['top_p'] = args['top_p']
    additional_params['generate_len'] = args['generate_len']
    data = globals()[dataset_name](data_path, tokenizer, **additional_params)

    sampler = torch.utils.data.SequentialSampler(data)
    iter_ = DataLoader(data, batch_size=args['batch_size'], collate_fn=data.collate, sampler=sampler)
    return data, iter_, sampler
