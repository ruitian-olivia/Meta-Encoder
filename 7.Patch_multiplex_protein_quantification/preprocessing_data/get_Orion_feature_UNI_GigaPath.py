import os
import sys
import json
import h5py
import timm
import torch
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from torchvision import transforms
from huggingface_hub import login, hf_hub_download
from pathlib import Path

parser = argparse.ArgumentParser(description='Model')
parser.add_argument('--model', type=str, default=None)

args = parser.parse_args()
model_name = args.model

device = 'cuda:0'

if(model_name == 'UNI'):
    print('==========USE MODEL: UNI ========')
    local_dir = "FM_models/FM_UNI/"
    os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
    model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
    model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    model = model.to(device)
    model.eval()
    
if(model_name == 'GigaPath'):
    print('==========USE MODEL: GigaPath ========')
    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)

    transform = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
    )
    model = model.to(device)
    model.eval()

ORION_CRC_root_path = '../Orion-CRC'
HE_path_df = pd.read_csv(os.path.join(ORION_CRC_root_path, 'ORION_HE_dataframe.csv'))
features_save_path = '../preprocessed_data/HE_features'

for _, row in HE_path_df.iterrows():
    HE_name = row['HE_path']
    HE_ID = os.path.splitext(HE_name)[0]

    HE_norm_path = os.path.join(ORION_CRC_root_path, 'ORION_dataset_20x_he_norm', HE_name)
    
    if os.path.exists(HE_norm_path):
        image = Image.open(HE_norm_path).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0)
        image = image.to('cuda:0', non_blocking=True)
        features = model(image)
        
        torch.save(features, os.path.join(features_save_path, model_name, f'{HE_ID}.pt'))
        
    else:
        print("File not exist:", HE_norm_path)
                
