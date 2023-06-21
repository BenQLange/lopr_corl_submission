import multiprocessing
import os
import pickle

import numpy as np
import torch
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

import sys
from src.models.encoders.autoencoder import Autoencoder
from src.util import instantiate_from_config
from copy import deepcopy

n_cpus = multiprocessing.cpu_count()
    
def load_dataset_from_config(config):
    data = instantiate_from_config(config['data'])    
    data.prepare_data()
    data.setup()
    return data

def save_sample(data, name):
    data_names = name.split('/')
    scene_name = data_names[-2]
    frame_name = data_names[-1].split('.')[0]
    save_path = os.path.join(split_path, scene_name)
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, frame_name + '.pkl')

    with open(save_path, 'wb') as f:
        pickle.dump(data.detach().cpu().numpy(), f)

if __name__ == '__main__':
    latent_path = None
    config_path = None
    ckpt_path = None
    data_path = None
    data_iterator = "src.data.dataset.Base"

    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    
    config['data']['params']['train']['params']['path'] = f'{data_path}/train'
    config['data']['params']['validation']['params']['path'] = f'{data_path}/val'

    config['data']['params']['test'] = deepcopy(config['data']['params']['validation'])
    config['data']['params']['test']['params']['path'] = f'{data_path}/test'

    ddconfig = config['model']['params']['ddconfig']
    config['model']['params']['lossconfig']['target'] = "torch.nn.Identity"
    config['data']['params']['train']['target'] = data_iterator
    config['data']['params']['validation']['target'] = data_iterator
    config['data']['params']['test']['target'] = data_iterator
    lossconfig = config['model']['params']['lossconfig']
    embed_dim = config['model']['params']['embed_dim']

    print(config)

    model = Autoencoder(ddconfig, lossconfig, embed_dim, ckpt_path=ckpt_path)
    model = model.cuda()
    model.eval()
    data = load_dataset_from_config(config)
    with torch.no_grad():
        for mode in ['train', 'val', 'test']: 
            print(f'Processing {mode} data')
            split_path = os.path.join(latent_path, mode)
            if mode == 'train':
                dataloader = data._train_dataloader() 
            elif mode  == 'val':
                dataloader = data._val_dataloader()
            elif mode == 'test':
                dataloader = data._test_dataloader()
            else:
                raise ValueError(f'Invalid mode {mode}')

            for batch in tqdm(dataloader):
                images = batch['image']
                images = images.to(model.device)
                images = images.permute(0, 3, 1, 2)
                filenames = batch['filename']
                posteriror = model.encode(images)
                params = posteriror.parameters
                params = torch.split(params, 1, dim=0)
                Parallel(n_jobs=n_cpus - 1)(delayed(save_sample)(param, filename) for param, filename in zip(params, filenames))


