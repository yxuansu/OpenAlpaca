from header import *

class DeepSpeedAgent:
    
    def __init__(self, model, args):
        super(DeepSpeedAgent, self).__init__()
        self.args = args
        self.model = model
        if self.args['mode'] == 'train':
            # load config parameters of deepspeed
            ds_params = json.load(open(self.args['ds_config_path']))
            ds_params['scheduler']['params']['total_num_steps'] = self.args['total_steps']
            ds_params['scheduler']['params']['warmup_num_steps'] = max(10, int(self.args['total_steps'] * self.args['warmup_rate']))
            print(f'[!] reset deepspeed training paramters:\n - total_steps: {self.args["total_steps"]}\n - warmup_steps: {ds_params["scheduler"]["params"]["warmup_num_steps"]}\n')
            self.ds_engine, self.optimizer, _ , _ = deepspeed.initialize(
                model=self.model, 
                model_parameters=self.model.parameters(),
                config_params=ds_params, 
                dist_init_required=True,
                args=types.SimpleNamespace(**args)
            )
            print(f'[!] init the deepspeed over')

    @torch.no_grad()
    def predict(self, batch):
        self.model.eval()
        string = self.model.generate_one_sample(batch)
        string = string.replace('</s>', '')
        return string

    def train_model(self, batch, current_step=0, pbar=None):
        self.ds_engine.module.train()
        loss, mle_acc = self.ds_engine(batch)

        self.ds_engine.backward(loss)
        self.ds_engine.step()
        pbar.set_description(f'[!] loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc*100, 2)}')
        pbar.update(1)
        if self.args['local_rank'] == 0 and current_step % self.args['logging_step'] == 0:
            elapsed = pbar.format_dict['elapsed']
            rate = pbar.format_dict['rate']
            remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0
            remaining = str(datetime.timedelta(seconds=remaining))
            logging.info(f'[!] progress: {round(pbar.n/pbar.total, 5)}; remaining time: {remaining}; loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc*100, 2)}')
            
        mle_acc *= 100
        return mle_acc
    
    def load_model(self, path, ckpt_id):
        self.ds_engine.load_checkpoint(path, ckpt_id, custom_load_fn=custom_load_fn, load_module_only=True)
        print(f'[!] load the latest model from {path} with ckpt id: {ckpt_id}')

    def save_model(self, path, current_step):
        ckpt_id = current_step
        self.ds_engine.save_checkpoint(path, ckpt_id)
        self.vocab.save_pretrained(f'{path}/openllama_hf')
        print(f'[!] save model into {path} with ckpt_id: {ckpt_id}')
