from header import *

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--model_path', default='/home/johnlan/pretrained/openalpaca', type=str)
    parser.add_argument('--max_shard_size', default='5GB', type=str)
    return parser.parse_args()

def main(model_path, max_shard_size):
    state_dict = torch.load(f'{model_path}/pytorch_model.bin')
    new_state_dict = OrderedDict()
    for key in state_dict:
        if key.startswith('model.'):
            new_key = key[6:]
        new_state_dict[new_key] = state_dict[key]
    config = LlamaConfig.from_pretrained(model_path)
    model = LlamaForCausalLM(config)
    model.load_state_dict(new_state_dict, strict=True)
    print(f'[!] load model over ...')
    model.save_pretrained(model_path, max_shard_size=max_shard_size)
    print(f'[!] make shards over ...')

    # delete the pytorch_model.bin and deepspeed checkpoints for saving disk memory
    os.remove(f'{model_path}/pytorch_model.bin')
    os.system(f'rm -rf {model_path}/0')
    os.system(f'rm -rf {model_path}/latest')
    os.system(f'rm -rf {model_path}/zero_to_fp32.py')
    #os.system(f'rm latest zero_to_fp32.py')

if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    main(args['model_path'], args['max_shard_size'])
