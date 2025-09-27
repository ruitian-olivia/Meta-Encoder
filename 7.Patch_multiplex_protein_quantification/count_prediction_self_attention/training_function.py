import os
import cv2
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train(model,train_loader,optimizer,poisson_loss,device):
    model.train()

    loss_all = 0
    for features, count in train_loader:
        
        features, count = features.to(device), count.to(device)

        optimizer.zero_grad()
        output = model(features)

        loss = poisson_loss(output, count)
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
    return loss_all / len(train_loader)

def valid(model,loader,poisson_loss,use_sampling=False):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_all = 0
    
    label = np.array([])
    pred = np.array([])
    for features, count in loader:
        features, count = features.to(device), count.to(device)
        output = model(features)
        
        
        loss = poisson_loss(output,count)
        loss_all += loss.item()
        
        if use_sampling:
            # 从泊松分布采样（随机性）
            predictions = torch.poisson(output)
        else:
            # 直接使用λ作为预测（期望值）
            predictions = torch.round(output)  # 四舍五入到最近整数
        
        _tmp_label = count.cpu().detach().numpy()
        _tmp_pred = predictions.cpu().detach().numpy()

        label = np.vstack([label,_tmp_label]) if label.size else _tmp_label
        pred = np.vstack([pred,_tmp_pred]) if pred.size else _tmp_pred

    return loss_all / len(loader), label, pred

def test(model,loader,use_sampling=False):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss = 0.  
    
    label = np.array([])
    pred = np.array([])
    x_coor = np.array([])
    y_coor = np.array([])
    for features, count in loader:
        features, count = features.to(device), count.to(device)
        output = model(features)
        
        if use_sampling:
            # 从泊松分布采样（随机性）
            predictions = torch.poisson(output)
        else:
            # 直接使用λ作为预测（期望值）
            predictions = torch.round(output)  # 四舍五入到最近整数
        
        _tmp_label = count.cpu().detach().numpy()
        _tmp_pred = predictions.cpu().detach().numpy()

        label = np.vstack([label,_tmp_label]) if label.size else _tmp_label
        pred = np.vstack([pred,_tmp_pred]) if pred.size else _tmp_pred

    return label, pred

def cal_gene_corr(label_df, pred_df, marker_list):
    pear_corr_list = []
    spea_corr_list = []
    
    for idx in range(len(marker_list)):
        label_gene = label_df[:,idx]
        pred_gene = pred_df[:,idx]
        
        pear_corr, _ = pearsonr(label_gene, pred_gene)
        spea_corr, _ = spearmanr(label_gene, pred_gene)
        
        pear_corr_list.append(pear_corr)
        spea_corr_list.append(spea_corr)
        
    result_dict = {"Pear_corr" : pear_corr_list,
                "Spea_corr" : spea_corr_list
                }
    result_df = pd.DataFrame(result_dict)
    result_df.index = marker_list
    
    return result_df

def save_count_result_df(label_df, pred_df, marker_list):
    print('label_df shape:', label_df.shape)
    print('pred_df shape:', pred_df.shape)
    
    label_array = np.asarray(label_df)
    pred_array = np.asarray(pred_df)

    # 获取样本数和marker数
    n_samples, n_markers = label_array.shape

    # 创建重复的marker名称数组
    marker_names = np.repeat(marker_list, n_samples)

    # 展平真实值和预测值数组
    true_values = label_array.ravel()
    pred_values = pred_array.ravel()

    # 创建DataFrame - 这是向量化的关键步骤
    result_df = pd.DataFrame({
        'Marker': marker_names,
        'True_Value': true_values,
        'Predicted_Value': pred_values
    })
    
    return result_df