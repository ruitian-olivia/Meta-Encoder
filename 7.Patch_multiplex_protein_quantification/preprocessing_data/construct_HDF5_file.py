import os
import h5py
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def preprocess_to_hdf5(marker_df, HE_root, encoder_list, marker_list, output_path):
    """
    将所有特征合并存储为HDF5文件，并保留encoder信息
    :param marker_df: 包含HE_path和其他信息的DataFrame
    :param HE_root: HE文件的根目录
    :param encoder_list: encoder名称列表
    :param marker_list: marker名称列表
    :param output_path: 输出HDF5文件路径
    """
    # 准备计数数据
    count_columns = [f"{name}_count" for name in marker_list]
    counts = marker_df[count_columns].values.astype("float32")
    HE_names_list = [name.encode('utf-8') for name in marker_df['HE_path'].values]
    encoder_names_list = [name.encode('utf-8') for name in encoder_list]
    
    with h5py.File(output_path, 'w') as hf:
        # 存储计数数据
        hf.create_dataset('counts', data=counts)
        
        # 存储metadata
        str_dtype = h5py.special_dtype(vlen=str)
        hf.create_dataset('HE_names', data=HE_names_list, dtype=str_dtype)
        
        # 创建encoder信息数据集
        hf.create_dataset('encoder_names', data=encoder_list, dtype=str_dtype)
        
        # 创建特征维度信息
        encoder_dims = {}
        sample_features = []
        
        # 收集每个encoder的特征维度和示例特征
        for encoder in encoder_list:
            path = os.path.join(HE_root, encoder, os.path.splitext(marker_df['HE_path'].iloc[0])[0] + '.pt')
            features = torch.load(path, map_location="cpu").detach().squeeze()
            encoder_dims[encoder] = features.shape[0]  # 记录每个encoder的特征维度
            sample_features.append(features)
        
        # 存储每个encoder的特征维度
        encoder_dim_array = np.array([encoder_dims[enc] for enc in encoder_list])
        hf.create_dataset('encoder_dimensions', data=encoder_dim_array)
        
        # 确定总特征维度
        total_features = sum(encoder_dims.values())
        feature_order = []  # 记录特征顺序
        
        # 创建特征顺序信息
        for encoder in encoder_list:
            feature_order.extend([f"{encoder}_feat_{i}" for i in range(encoder_dims[encoder])])
        
        feature_order_list = [name.encode('utf-8') for name in feature_order]

        print("feature_order_list:", feature_order_list)
        
        hf.create_dataset('feature_order', data=feature_order_list, dtype=str_dtype)
        
        # 创建可扩展的dataset
        features_dset = hf.create_dataset(
            'features',
            shape=(len(marker_df), total_features),
            dtype='float32',
            chunks=(1, total_features),  # 优化小批量读取
            compression='gzip'
        )
        
        # 逐个样本处理
        for idx, HE_name in enumerate(tqdm(marker_df['HE_path'].values)):
            features_list = []
            for encoder in encoder_list:
                path = os.path.join(HE_root, encoder, os.path.splitext(HE_name)[0] + '.pt')
                features = torch.load(path, map_location="cpu").detach().squeeze()
                features_list.append(features)
            
            merged_features = torch.cat(features_list, dim=0)
            features_dset[idx] = merged_features.numpy()
         
ORION_CRC_root_path = '../Orion-CRC'
preprocessed_saved_path = '../preprocessed_data/HE_features'
HE_path_df = pd.read_csv(os.path.join(ORION_CRC_root_path, 'ORION_HE_dataframe.csv'))
saved_root_path = '../preprocessed_data/h5_files'

patient_ID_list = HE_path_df['orion_slide_id'].unique().tolist()

test_list = patient_ID_list[-8:]
print("Test set patient_ID:", test_list)

marker_list = ['CD31', 'CD45', 'CD68', 'CD4',
       'FOXP3', 'CD8a', 'CD45RO', 'CD20',
       'PD-L1', 'CD3e', 'CD163', 'E-cadherin',
       'Ki67', 'Pan-CK', 'SMA']

test_df = HE_path_df[HE_path_df['orion_slide_id'].isin(test_list)]
train_val_df = HE_path_df[~HE_path_df['orion_slide_id'].isin(test_list)]

train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)

preprocess_to_hdf5(
    marker_df=train_df,
    HE_root=preprocessed_saved_path,
    encoder_list=["CHIEF", "GigaPath", "UNI"],
    marker_list=marker_list,
    output_path=os.path.join(saved_root_path, 'train_data.h5')
)

preprocess_to_hdf5(
    marker_df=val_df,
    HE_root=preprocessed_saved_path,
    encoder_list=["CHIEF", "GigaPath", "UNI"],
    marker_list=marker_list,
    output_path=os.path.join(saved_root_path, 'val_data.h5')
)

preprocess_to_hdf5(
    marker_df=test_df,
    HE_root=preprocessed_saved_path,
    encoder_list=["CHIEF", "GigaPath", "UNI"],
    marker_list=marker_list,
    output_path=os.path.join(saved_root_path, 'test_data.h5')
)

