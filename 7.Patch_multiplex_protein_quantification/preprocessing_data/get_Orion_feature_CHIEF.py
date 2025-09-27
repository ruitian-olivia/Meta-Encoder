import os
import json
import h5py
import sys
import torch
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path

sys.path.append('./FM_models/FM_CHIEF/')
from models.ctran import ctranspath


device = 'cuda:0'

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
        
        torch.save(features, os.path.join(features_save_path, 'CHIEF', f'{HE_ID}.pt'))
        
    else:
        print("File not exist:", HE_norm_path)
        
