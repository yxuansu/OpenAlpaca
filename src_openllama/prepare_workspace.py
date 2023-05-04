from config import load_base_config
import os

if __name__ == "__main__":
    config = load_base_config()
    models = config['models']
    root_dir = config['root_dir']

    for folder in ['rest', 'ckpt']:
        path = f'{root_dir}/{folder}'
        if not os.path.exists(path):
            os.mkdir(path)
        path = f'{root_dir}/{folder}'
        if not os.path.exists(path):
            os.mkdir(path)
        for model in models:
            path = f'{root_dir}/{folder}/{model}'
            if not os.path.exists(path):
                os.mkdir(path)
    print(f'[!] init the folder under the {root_dir} over')
