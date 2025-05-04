import os
import json
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

task_list = ['CCRCC',
             'COAD',
             'HCC',
             'IDC',
             'LUNG',
             'LYMPH_IDC',
             'PAAD',
             'PRAD',
             'READ',
             'SKCM']

bench_data_root = './hest-bench'
UNI_features_root = './embedding_features/UNI'
CHIEF_features_root = './embedding_features/CHIEF'
GigaPath_features_root = './embedding_features/GigaPath'
merged_features_root = './embedding_features/merged'

for task_name in task_list:
    print('task_name:', task_name)
    train_split = f'./hest-bench/{task_name}/splits/train_0.csv'
    test_split = f'./hest-bench/{task_name}/splits/test_0.csv'
    train_df = pd.read_csv(train_split)
    test_df = pd.read_csv(test_split)
    
    for split in [train_df, test_df]:
        for i in tqdm(range(len(split))):
            sample_id = split.iloc[i]['sample_id']
            print('sample_id:', sample_id)
            
            merged_features_path = os.path.join(merged_features_root, task_name)
            if not os.path.exists(merged_features_path):
                os.makedirs(merged_features_path)
    
            UNI_features_path = os.path.join(UNI_features_root, task_name, sample_id+'.h5')
            assert os.path.isfile(UNI_features_path)
            CHIEF_features_path = os.path.join(CHIEF_features_root, task_name, sample_id+'.h5')
            assert os.path.isfile(CHIEF_features_path)
            GigaPath_features_path = os.path.join(GigaPath_features_root, task_name, sample_id+'.h5')
            assert os.path.isfile(GigaPath_features_path)
            
            with h5py.File(os.path.join(merged_features_path, sample_id+'.h5'), 'a') as f_merged:
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

