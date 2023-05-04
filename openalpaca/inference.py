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
    agent = load_model(args)

    # instruction = 'What is the Natural Language Processing'
    instruction = input('Input your instruction:')
    prompt_no_input = f'### Instruction:\n{instruction}\n\n### Response:'
    tokens = agent.tokenizer.encode(prompt_no_input)
    tokens = [1] + tokens + [2] + [1]
    tokens = torch.LongTensor(tokens[-1024:]).unsqueeze(0)

    instance = {
        'input_ids': tokens,
        'decoding_method': 'sampling',
        'top_k': 50,
        'top_p': 0.9,
        'generate_len': 256
    }
    rest = agent.predict(instance)
    print(f'[!] Generation results: {rest}')

if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    main(**args)
