from header import *

class OpenLLAMAModel(nn.Module):

    def __init__(self, **args):
        super(OpenLLAMAModel, self).__init__()
        self.args = args
        model = args['pretrained_model']
        self.vocab = LlamaTokenizer.from_pretrained(args['tokenizer'])
        self.vocab.bos_token_id = 1
        self.vocab.eos_token_id = 2
        self.vocab.pad_token_id = 2
        self.pad = 2

        if self.args['mode'] == 'train':
            self.model = LlamaForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
        else:
            path = f'{self.args["root_dir"]}/ckpt/openllama/openllama_hf'
            self.model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.float16)
        self.vocab_size = self.model.config.vocab_size
        self.model.cuda(torch.cuda.current_device())
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad, reduction='none')
        self.ppl_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        total = sum([param.nelement() for param in self.parameters()])
        print('[!] Model Size: %2fB' % (total/1e9))

    @torch.no_grad()
    def generate_one_sample(self, batch):
        self.model.eval()
        ids = batch['input_ids'].cuda()
        length = len(ids[0])
        model = self.model
        if batch['decoding_method'] == 'greedy':
            output = model.generate(
                input_ids=ids, 
                max_length=length + batch['generate_len'], 
                use_cache=True
            )
        elif batch['decoding_method'] == 'sampling':
            output = model.generate(
                input_ids=ids, 
                do_sample=True,
                top_p=batch['top_p'],
                top_k=batch['top_k'],
                max_length=length + batch['generate_len'], 
                use_cache=True
            )
        else:
            raise Exception(f'[!] Unknow generate method: {batch["decoding_method"]}')

        output = output[0][length:]
        string = self.vocab.decode(output, skip_special_tokens=False)
        string = string.replace('<s>', '').replace('</s>', '')
        return string.strip()

    def forward(self, batch):
        inputs = {}
        for key, value in batch.items():
            try:
                value = value.cuda()
                inputs[key] = value
            except:
                continue

        outputs = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['labels'])
        loss = outputs.loss
        logits = outputs.logits[:, :-1, :]
        labels = inputs['labels'][:, 1:]
        
        # calculate the token accuarcy
        chosen_tokens = torch.max(logits, dim=-1)[1]    # [B, S-1]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)    # [B*S]
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask    # [B*S]
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()
        return loss, gen_acc
 
