import os
import json
import h5py
import torch
import numpy as np
import pandas as pd
import scanpy as sc
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn as nn
import sys

import os
import torch
from torchvision import transforms

import os
import argparse
from pathlib import Path

import sys

sys.path.append('./FM_models/FM_CHIEF/')
from models.ctran import ctranspath


parser = argparse.ArgumentParser(description='Model')
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--bench_data_root', type=str, default=None)
parser.add_argument('--save_h5_root', type=str, default=None)

args = parser.parse_args()
model_name = args.model
bench_data_root = args.bench_data_root
save_h5_root = args.save_h5_root

device = 'cuda:0'

def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        
        if key not in file:
            
            data_type = val.dtype
            
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path

class H5HESTDataset(Dataset):
    """ Dataset to read ST + H&E from .h5 """
    def __init__(self, h5_path, img_transform=None, chunk_size=1000):
        self.h5_path = h5_path
        self.img_transform = img_transform
        self.chunk_size = chunk_size
        with h5py.File(h5_path, 'r') as f:
            self.n_chunks = int(np.ceil(len(f['barcode']) / chunk_size))
        
    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start_idx = idx * self.chunk_size
        end_idx = (idx + 1) * self.chunk_size
        with h5py.File(self.h5_path, 'r') as f:
            imgs = f['img'][start_idx:end_idx]
            barcodes = f['barcode'][start_idx:end_idx].flatten().tolist()
            coords = f['coords'][start_idx:end_idx]
        if self.img_transform:
            imgs = torch.stack([self.img_transform(Image.fromarray(img)) for img in imgs])
        return {'imgs': imgs, 'barcodes': barcodes, 'coords': coords}

if(model_name == 'CHIEF'):
    print('==========USE MODEL: CHIEF ========')
    model = ctranspath()
    model.head = nn.Identity()
    td = torch.load('./FM_models/FM_CHIEF/model_weight/CHIEF_CTransPath.pth')
    model.load_state_dict(td['model'], strict=True)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = mean, std = std)
        ]
    )
    model = model.to(device)
    model.eval()

print_every = 10
save_h5_path = os.path.join(save_h5_root, model_name)

if not os.path.exists(save_h5_path):
    os.makedirs(save_h5_path)

SampleList = os.listdir(bench_data_root)
for i in tqdm(range(len(SampleList))):

    sample_id = SampleList[i]
    tile_h5_path = os.path.join(bench_data_root,sample_id,'patches.h5')

    assert os.path.isfile(tile_h5_path)
    tile_dataset = H5HESTDataset(tile_h5_path,img_transform=transform,chunk_size=1200)
    
    expr_h5ad_path = os.path.join(bench_data_root,sample_id, 'aligned_adata.h5ad')
    assert os.path.isfile(expr_h5ad_path)

    tile_dataloader = torch.utils.data.DataLoader(tile_dataset, 
                                    batch_size=1, 
                                    shuffle=False)
    mode = 'w'
    output_path = os.path.join(save_h5_path, sample_id+'.h5')
    for count, batch in tqdm(enumerate(tile_dataloader), total=len(tile_dataloader)):
        imgs = torch.tensor(batch['imgs'].squeeze(0).numpy(),dtype=torch.float32)
        coords = batch['coords'].numpy().squeeze(0)
        barcodes = np.array(batch['barcodes'])
        with torch.no_grad():    
            if count % print_every == 0:
                print('batch {}/{}, {} files processed'.format(count, len(tile_dataloader), count))
            imgs = imgs.to('cuda:0', non_blocking=True)
            # imgs = imgs.permute(0, 2,3,1)
            if(model_name == 'Virchow2'):
                output = model(imgs)
                class_token = output[:, 0]    # size: 1 x 1280
                patch_tokens = output[:, 5:]  # size: 1 x 256 x 1280, tokens 1-4 are register tokens so we ignore those
                features = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  
            else:
                features = model(imgs)
            features = features.cpu().numpy()     
            asset_dict = {'features': features, 'coords': coords,'barcodes':barcodes}
            save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
            mode = 'a'