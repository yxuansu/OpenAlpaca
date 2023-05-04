from header import *
from datasets import *
from model import *
from config import *
from train_sft import *


def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--model', type=str)
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--test_base_model', type=str, default='False')
    return parser.parse_args()

def main(**args):
    args['mode'] = 'test'
    config = load_config(args)
    args.update(config)
    initialize_distributed(args)
    set_random_seed(args['seed'])
    test_data, test_iter, _ = load_sft_test_dataset(args)
    agent = load_model(args)
    
    torch.distributed.barrier()

    results = []
    for batch in tqdm(test_iter):
        rest = agent.predict(batch)
        batch['generation'] = rest
        try:
            batch.pop('input_ids')
        except:
            pass
        results.append(batch)
        pprint.pprint(batch)

    save_path = f'{args["root_dir"]}/rest/sft/{args["model"]}/test_eval_gen_{args["test_base_model"]}.txt'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    main(**args)
