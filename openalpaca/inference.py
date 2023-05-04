from header import *

def main():
    model_path = '/home/johnlan/pretrained_models/openllama'
    model = LlamaForCausalLM.from_pretrained(model_path).cuda()
    tokenizer = LlamaTokenizer.from_pretrained(model_path)

    # instruction = 'What is the Natural Language Processing'
    instruction = input('Input your instruction: ')
    prompt_no_input = f'### Instruction:\n{instruction}\n\n### Response:'
    tokens = tokenizer.encode(prompt_no_input)
    tokens = [1] + tokens + [2] + [1]
    tokens = torch.LongTensor(tokens[-1024:]).unsqueeze(0).cuda()

    instance = {
        'input_ids': tokens,
        'top_k': 50,
        'top_p': 0.9,
        'generate_len': 256
    }
    length = len(tokens[0])
    rest = model.generate(
        input_ids=tokens, 
        max_length=length+instance['generate_len'], 
        use_cache=True, 
        do_sample=True, 
        top_p=instance['top_p'], 
        top_k=instance['top_k']
    )
    output = rest[0][length:]
    string = tokenizer.decode(output, skip_special_tokens=False)
    string = string.replace('<s>', '').replace('</s>', '').strip()
    print(f'[!] Generation results: {string}')

if __name__ == "__main__":
    main()
