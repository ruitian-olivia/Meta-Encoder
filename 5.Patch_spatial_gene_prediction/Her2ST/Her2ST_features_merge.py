import os
import json
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

Her2ST_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

merged_features_root = './embedding_features/merged'

for patient_name in Her2ST_list:
    print('patient_name:', patient_name)
    UNI_features_root = f'./embedding_features/UNI/{patient_name}'
    CHIEF_features_root = f'./embedding_features/CHIEF/{patient_name}'
    GigaPath_features_root = f'./embedding_features/GigaPath/{patient_name}'
    
    merged_features_path = os.path.join(merged_features_root, 'Her2ST_'+patient_name)
    if not os.path.exists(merged_features_path):
        os.makedirs(merged_features_path)
    
    tissue_list = os.listdir(UNI_features_root)
    for tissue_name in tissue_list:
        print('tissue_name:', tissue_name)
        
        UNI_features_path = os.path.join(UNI_features_root, tissue_name)
        assert os.path.isfile(UNI_features_path)
        CHIEF_features_path = os.path.join(CHIEF_features_root, tissue_name)
        assert os.path.isfile(CHIEF_features_path)
        GigaPath_features_path = os.path.join(GigaPath_features_root, tissue_name)
        assert os.path.isfile(GigaPath_features_path)
        
        with h5py.File(os.path.join(merged_features_path, tissue_name), 'a') as f_merged:
            with h5py.File(UNI_features_path, 'r') as f_UNI:
                barcodes = f_UNI['barcodes'][:] 
                f_merged.create_dataset('barcodes', data=barcodes)
                coords = f_UNI['coords'][:] 
                f_merged.create_dataset('coords', data=coords)
                UNI_features = f_UNI['features'][:] 
                f_merged.create_dataset('UNI_features', data=UNI_features)
            with h5py.File(CHIEF_features_path, 'r') as f_CHIEF:
                CHIEF_features = f_CHIEF['features'][:] 
                f_merged.create_dataset('CHIEF_features', data=CHIEF_features)
            with h5py.File(GigaPath_features_path, 'r') as f_GigaPath:
                GigaPath_features = f_GigaPath['features'][:] 
                f_merged.create_dataset('GigaPath_features', data=GigaPath_features)
                