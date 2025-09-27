import os
import cv2
import sys
import json
import time
import h5py
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.cm as cm
from torchvision import models
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from model_function import protein_poisson_count
from training_function import setup_seed, train, valid, test, cal_gene_corr, save_count_result_df
from pytorchtools import EarlyStopping

class HDF5MarkerDataset(Dataset):
    def __init__(self, hdf5_path):

        super(HDF5MarkerDataset, self).__init__()
        self.hdf5_path = hdf5_path
        
        with h5py.File(hdf5_path, 'r') as hf:
            self.num_samples = hf['features'].shape[0]
            self.encoder_names = [name.decode() if isinstance(name, bytes) else name 
                                for name in hf['encoder_names'][:]]
            self.encoder_dims = {name: dim for name, dim in 
                               zip(self.encoder_names, hf['encoder_dimensions'][:])}
            self.feature_order = [name.decode() if isinstance(name, bytes) else name 
                                for name in hf['feature_order'][:]]
            
            self.feature_ranges = {}
            start_idx = 0
            for enc in self.encoder_names:
                end_idx = start_idx + self.encoder_dims[enc]
                self.feature_ranges[enc] = (start_idx, end_idx)
                start_idx = end_idx
        
        self.hdf5_file = None
    
    def get_encoder_info(self):
        return {
            'encoder_names': self.encoder_names,
            'encoder_dims': self.encoder_dims,
            'feature_ranges': self.feature_ranges,
            'feature_order': self.feature_order
        }
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx, encoder=None):

        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.hdf5_path, 'r')
        
        counts = torch.from_numpy(self.hdf5_file['counts'][idx])
        
        if encoder is None:
            features = torch.from_numpy(self.hdf5_file['features'][idx])
        else:
            if encoder not in self.feature_ranges:
                raise ValueError(f"Unknown encoder: {encoder}. Available encoders: {self.encoder_names}")
            
            start, end = self.feature_ranges[encoder]
            features = torch.from_numpy(self.hdf5_file['features'][idx, start:end])
        
        return features, counts
    
    def get_features_by_encoder(self, idx, encoder):
        return self.__getitem__(idx, encoder=encoder)
    
    def __del__(self):
        if self.hdf5_file is not None:
            self.hdf5_file.close()
    
def poisson_loss(pred, target):
    return torch.mean(pred - target * torch.log(pred + 1e-8))

def split_list(lst, sizes):
    result = []
    start = 0
    for size in sizes:
        end = start + size
        result.append(lst[start:end])
        start = end
    return result

# model training arg parser
parser = argparse.ArgumentParser(description="Arguments for model training.")

parser.add_argument(
    "--model_name",
    type=str,
    help="The name of the trainned model",
)
parser.add_argument(
    '--encoder_list', 
    type=str, 
    nargs='+', 
    help='List of encoder names to concate'
)
parser.add_argument(
    "--embed_dim",
    type=int,
    help="Embedding dimension",
)
parser.add_argument(
    "--num_heads",
    type=int,
    help="The number of attention heads",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    help="Learning rate",
)
parser.add_argument(
    "--weight_decay", 
    type=float, 
    help="Weight decay"
)

parser.add_argument(
    "--epochs",
    type=int,
    help="The number of epochs",
)
parser.add_argument(
    "--patience",
    type=int,
    help="The number of patience",
)
parser.add_argument(
    "--batch_size",
    type=int,
    help="The number of batch size",
)
parser.add_argument(
    "--seed_num",
    default=42,
    type=int,
    help="Seed number",
)
args = parser.parse_args()

try:
    model_name = args.model_name
    encoder_list = args.encoder_list
    embed_dim = args.embed_dim
    num_heads = args.num_heads
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    epochs = args.epochs
    patience = args.patience
    batch_size = args.batch_size
    seed_num = args.seed_num
except:
    print("error in parsing args")

setup_seed(seed_num)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: {}'.format(device))

if device=='cpu':
    sys.exit(0)
    
ORION_CRC_root_path = '../Orion-CRC'
hdf5_root_path = '../preprocessed_data/h5_files'

marker_list = ['CD31', 'CD45', 'CD68', 'CD4',
       'FOXP3', 'CD8a', 'CD45RO', 'CD20',
       'PD-L1', 'CD3e', 'CD163', 'E-cadherin',
       'Ki67', 'Pan-CK', 'SMA']

num_marker = len(marker_list)
print("num_marker:", num_marker)

num_feature_dict = {'CHIEF': 768, 'UNI': 1024, 'GigaPath': 1536}
num_feature = 0
for encoder in encoder_list:
    num_feature += num_feature_dict[encoder]

runs = 10
train_loss = np.zeros((epochs))
val_loss = np.zeros((epochs))
val_pear_corr = np.zeros((epochs))
val_spea_corr = np.zeros((epochs))
min_loss = 1e10

test_loss = np.zeros(runs)
result_df_list = []

model_save_dir = os.path.join("model_weights", model_name)
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

result_save_dir = os.path.join("model_result", model_name)
if not os.path.exists(result_save_dir):
    os.makedirs(result_save_dir)    

epoch_counter = 0

train_dataset = HDF5MarkerDataset(os.path.join(hdf5_root_path, 'train_data.h5'))
val_dataset = HDF5MarkerDataset(os.path.join(hdf5_root_path, 'val_data.h5'))
test_dataset = HDF5MarkerDataset(os.path.join(hdf5_root_path, 'test_data.h5'))
print("len(train_dataset):", len(train_dataset))
print("len(val_dataset):", len(val_dataset))
print("len(test_dataset):", len(test_dataset))

num_workers = 2
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
val_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=num_workers)
    
model = protein_poisson_count(embed_dim, num_heads, num_feature, num_marker).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

early_stopping = EarlyStopping(patience=patience, verbose=True, path="{}/model_{}.pth".format(model_save_dir,model_name))

for epoch in range(epochs):
    print("epoch:", epoch)
    loss = train(model,train_loader,optimizer,poisson_loss,device) 
    print("loss:", loss)
    
    train_loss[epoch] = loss
    val_loss[epoch], val_label, val_pred = valid(model,val_loader,poisson_loss,use_sampling=True)
    
    val_result_df = cal_gene_corr(val_label, val_pred, marker_list)
    val_pear_corr[epoch] = val_result_df["Pear_corr"].mean()
    val_spea_corr[epoch] = val_result_df["Spea_corr"].mean()

    epoch_counter += 1
    print("Epoch: {:03d}, \
            Train loss: {:.5f}, Val loss: {:.5f}, \
            Val pearson corr: {:.5f}, Val spearman corr: {:.5f}"\
            .format(epoch+1,\
                train_loss[epoch], val_loss[epoch],\
                val_pear_corr[epoch], val_spea_corr[epoch]))
    
    early_stopping(val_loss[epoch], model)
    if early_stopping.early_stop:
        print("Early stopping")
        break
    if val_loss[epoch] < min_loss:
        min_loss = val_loss[epoch]
                
model.load_state_dict(torch.load("{}/model_{}.pth".format(model_save_dir,model_name)))

for run_index in range(runs):
    print("RUN index:", run_index)
    test_label, test_pred = test(model,test_loader,use_sampling=True)

    print("test_label:", test_label)
    print("test_pred:", test_pred)

    test_result_df = cal_gene_corr(test_label, test_pred, marker_list)
    test_result_df.to_csv(os.path.join(result_save_dir,'Test_index_{}_result.csv'.format(run_index)), float_format='%.4f')
    result_df_list.append(test_result_df)
    
    test_count_result_df = save_count_result_df(test_label, test_pred, marker_list)
    test_count_result_df.to_csv(os.path.join(result_save_dir,'Test_index_{}_count_result.csv'.format(run_index)),index=False,float_format='%.0f')
    
result_df_concat = pd.concat(result_df_list)
by_row_index = result_df_concat.groupby(result_df_concat.index)
result_df_means = by_row_index.mean()
result_df_std = by_row_index.std()
result_df_std.columns=['Pear_corr_std', 'Spea_corr_std']
result_df_all = pd.concat([result_df_means, result_df_std], axis=1)
pear_result_df = result_df_all.loc[:, ('Pear_corr', 'Pear_corr_std')]
pear_result_df.sort_values(by="Pear_corr", inplace=True, ascending=False)
pear_result_df.to_csv(os.path.join(result_save_dir,'result_df_Pearson.csv'), float_format='%.4f')
spea_result_df = result_df_all.loc[:, ('Spea_corr', 'Spea_corr_std')]
spea_result_df.sort_values(by="Spea_corr", inplace=True, ascending=False)
spea_result_df.to_csv(os.path.join(result_save_dir,'result_df_Spearman.csv'), float_format='%.4f')

result_mean = result_df_all.mean(axis=0)
result_median = result_df_all.median(axis=0)
print("------------Summary------------")
print("Result Mean:\n", result_mean)
print("Result Median:\n", result_median)
 