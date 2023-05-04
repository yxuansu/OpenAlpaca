from header import *
from datasets import *
from model import *
from config import *


def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--model', type=str)
    parser.add_argument('--data_path', default='../data/dolly.json', type=str)
    parser.add_argument('--local_rank', default=0, type=int)
    return parser.parse_args()

def initialize_distributed(args):
    args['master_ip'] = os.getenv('MASTER_ADDR', 'localhost')
    args['master_port'] = os.getenv('MASTER_PORT', '6000')
    args['world_size'] = int(os.getenv('WORLD_SIZE', '1'))
    args['local_rank'] = int(os.getenv('RANK', '0')) % torch.cuda.device_count()
    device = args['local_rank'] % torch.cuda.device_count()
    torch.cuda.set_device(device)
    deepspeed.init_distributed(dist_backend='nccl')

def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

def config_env(args):
    args['root_dir'] = '../'
    args['mode'] = 'train'
    config = load_config(args)
    args.update(config)
    initialize_distributed(args)
    set_random_seed(args['seed'])

def main(**args):
    config_env(args)
    args['ds_config_path'] = f'dsconfig/{args["model"]}.json'
    dschf = HfDeepSpeedConfig(args['ds_config_path'])
    args['dschf'] = dschf

    logging.basicConfig(
        format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s', 
        level=logging.DEBUG,
        filename=f'{args["root_dir"]}/rest/{args["model"]}/train_{time.asctime()}.log',
        filemode='a'
    )
    
    train_data, train_iter, sampler = load_sft_dataset(args)

    length = args['epochs'] * len(train_data) // args['world_size'] // dschf.config['train_micro_batch_size_per_gpu']
    total_steps = args['epochs'] * len(train_data) // dschf.config['train_batch_size']
    args['total_steps'] = total_steps
    agent = load_model(args)
    torch.distributed.barrier()
    
    pretrained_model_name = args['pretrained_model'].replace('/', '_')

    save_path = f'{args["root_dir"]}/ckpt/{args["model"]}'
    torch.distributed.barrier()

    pbar = tqdm(total=length)    # maximum total number
    save_counter = 0
    current_step = 0

    for epoch_i in tqdm(range(args['epochs'])):
        for batch in train_iter:
            agent.train_model(
                batch, 
                current_step=current_step, 
                pbar=pbar
            )
            current_step += 1
        agent.save_model(save_path, save_counter)
        save_counter += 1

if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    main(**args)
