import json
from datasets import load_dataset

def process_item(item):
    res_dict = {'instruction':item['instruction'],
               'input':item['context'],
               'output':item['response']}
    return res_dict

def get_combine_length(item):
    instruction_len = len(item['instruction'].split())
    input_len = len(item['input'].split())
    output_len = len(item['output'].split())
    return instruction_len + input_len + output_len

if __name__ == '__main__':
    dataset = load_dataset("databricks/databricks-dolly-15k")
    res_list = []
    for item in dataset['train']:
        one_item = process_item(item)
        if get_combine_length(one_item) <= 2000:
           res_list.append(one_item)
        else:
            # remove instance that is too long
            pass

    save_path = r'./openalpaca.json'
    with open(save_path, 'w') as outfile:
        json.dump(res_list, outfile, indent=4)
