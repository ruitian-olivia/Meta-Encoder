import os
import json
import h5py
import glob
import torch
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import torch.utils.tensorboard as tensorboard
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr

def read_assets_from_h5(h5_path, keys=None, skip_attrs=False, skip_assets=False):
    assets = {}
    attrs = {}
    with h5py.File(h5_path, 'r') as f:
        if keys is None:
            keys = list(f.keys())
        for key in keys:
            if not skip_assets:
                assets[key] = f[key][:]
            if not skip_attrs:
                if f[key].attrs is not None:
                    attrs[key] = dict(f[key].attrs)
    return assets, attrs

def normalize_adata(adata: sc.AnnData, scale=1e6, smooth=False) -> sc.AnnData: # type: ignore
    """
    Normalize each spot by total gene counts + Logarithmize each spot
    """
    
    import scanpy as sc
    
    filtered_adata = adata.copy()
    
    if smooth:
        adata_df = adata.to_df()
        for index, df_row in adata.obs.iterrows():
            row = int(df_row['array_row'])
            col = int(df_row['array_col'])
            neighbors_index = adata.obs[((adata.obs['array_row'] >= row - 1) & (adata.obs['array_row'] <= row + 1)) & \
                ((adata.obs['array_col'] >= col - 1) & (adata.obs['array_col'] <= col + 1))].index
            neighbors = adata_df.loc[neighbors_index]
            nb_neighbors = len(neighbors)
            
            avg = neighbors.sum() / nb_neighbors
            filtered_adata[index] = avg
    
    sc.pp.normalize_total(filtered_adata, target_sum=1, inplace=True)
    # Facilitate training when using the MSE loss. 
    # This'trick' is also used by Y. Zeng et al in "Spatial transcriptomics prediction from histology jointly through Transformer and graph neural networks"
    filtered_adata.X = filtered_adata.X * scale 
    
    # Logarithm of the expression
    sc.pp.log1p(filtered_adata) 

    return filtered_adata

def load_adata(expr_path, genes = None, barcodes = None, normalize=False):
    adata = sc.read_h5ad(expr_path)
    if normalize:
        adata = normalize_adata(adata)
    if barcodes is not None:
        adata = adata[barcodes]
    if genes is not None:
        adata = adata[:, genes]

    return adata.to_df()

def merge_dict(main_dict, new_dict, value_fn = None):
    """
    Merge new_dict into main_dict. If a key exists in both dicts, the values are appended. 
    Else, the key-value pair is added.
    Expects value to be an array or list - if not, it is converted to a list.
    If value_fn is not None, it is applied to each item in each value in new_dict before merging.
    Args:
        main_dict: main dict
        new_dict: new dict
        value_fn: function to apply to each item in each value in new_dict before merging
    """
    if value_fn is None:
        value_fn = lambda x: x
    for key, value in new_dict.items():
        if not isinstance(value, list):
            value = [value]
        value = [value_fn(v) for v in value]
        if key in main_dict:
            main_dict[key] = main_dict[key] + value
        else:
            main_dict[key] = value
    return main_dict

class HESTDataset(Dataset):
    def __init__(self, UNI_features, CHIEF_features, GigaPath_features, gene_expression):
        self.UNI_features = UNI_features
        self.CHIEF_features = CHIEF_features
        self.GigaPath_features = GigaPath_features
        self.gene_expression = gene_expression

    def __len__(self):
        return len(self.gene_expression)

    def __getitem__(self, idx):
        UNI_feature_item = self.UNI_features[idx]
        CHIEF_feature_item = self.CHIEF_features[idx]
        GigaPath_feature_item = self.GigaPath_features[idx]
        gene_expression_item = self.gene_expression[idx]
        
        return UNI_feature_item, CHIEF_feature_item, GigaPath_feature_item, gene_expression_item

class LinearProbeReg(torch.nn.Module):

    def __init__(self, embed_dim: int = 1536, gene_num: int = 10):
        super(LinearProbeReg, self).__init__()

        self.fc = torch.nn.Linear(embed_dim, gene_num)

    def forward(self, x):
        return self.fc(x)

def evaluate(model, criterion, val_loader, device, genes, args):
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        total_loss = 0

        pred_gather, targets_gather = None, None
        for _, batch in enumerate(val_loader):
            UNI_features, CHIEF_features, GigaPath_features, targets = batch
            UNI_features, CHIEF_features, GigaPath_features, targets = UNI_features.to(device), CHIEF_features.to(device), GigaPath_features.to(device), targets.to(device)

            if args.feature_type == 'UNI':
                output = model(UNI_features)
            elif args.feature_type == 'CHIEF':
                output = model(CHIEF_features)
            elif args.feature_type == 'GigaPath':
                output = model(GigaPath_features)
            elif args.feature_type == 'Concat':
                concat_features = torch.cat((UNI_features, CHIEF_features, GigaPath_features), dim=1)
                output = model(concat_features)

            loss = criterion(output, targets)
            total_loss += loss.item()
            
            if pred_gather is None:
                pred_gather = output.cpu().numpy()
                targets_gather = targets.cpu().numpy()
            else:
                pred_gather = np.concatenate((pred_gather, output.cpu().numpy()), axis=0)
                targets_gather = np.concatenate((targets_gather, targets.cpu().numpy()), axis=0)
    
    val_loss = total_loss / len(val_loader)
    
    errors = []
    r2_scores = []
    pearson_corrs = []
    pearson_genes = []
    for target_gene in range(targets_gather.shape[1]):
        pred_value = pred_gather[:, target_gene]
        targets_value = targets_gather[:, target_gene]
        l2_error = float(np.mean((pred_value - targets_value)**2))
        r2_score = float(1 - np.sum((targets_value - pred_value)**2) / np.sum((targets_value - np.mean(targets_value))**2))
        pearson_corr, _ = pearsonr(targets_value, pred_value)
        if np.isnan(pearson_corr):
            print(targets_value)
            print(pred_value)
        errors.append(l2_error)
        r2_scores.append(r2_score)
        pearson_corrs.append(pearson_corr)
        score_dict = {
            'name': genes[target_gene],
            'pearson_corr': pearson_corr,
        }
        pearson_genes.append(score_dict)
    
    results = {'loss': val_loss,
            'l2_errors': list(errors), 
            'r2_scores': list(r2_scores),
            'pearson_corrs': pearson_genes,
            'pearson_mean': float(np.mean(pearson_corrs)),
            'pearson_std': float(np.std(pearson_corrs)),
            'l2_error_q1': float(np.percentile(errors, 25)),
            'l2_error_q2': float(np.median(errors)),
            'l2_error_q3': float(np.percentile(errors, 75)),
            'r2_score_q1': float(np.percentile(r2_scores, 25)),
            'r2_score_q2': float(np.median(r2_scores)),
            'r2_score_q3': float(np.percentile(r2_scores, 75))}
    
    return results
    
def train_eval(model,
          train_loader,
          output_dir,
          genes,
          args):

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # set Tensorboard
    tensorboard_dir = os.path.join(output_dir, 'tensorboard')
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = tensorboard.SummaryWriter(tensorboard_dir)
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Set the learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_num*len(train_loader) // args.n_cycles , eta_min=args.min_lr)
    
    for epoch in range(args.epochs_num):
        for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs_num}"):
            UNI_features, CHIEF_features, GigaPath_features, targets = batch
            UNI_features, CHIEF_features, GigaPath_features, targets = UNI_features.to(device), CHIEF_features.to(device), GigaPath_features.to(device), targets.to(device)
            
            if args.feature_type == 'UNI':
                output = model(UNI_features)
            elif args.feature_type == 'CHIEF':
                output = model(CHIEF_features)
            elif args.feature_type == 'GigaPath':
                output = model(GigaPath_features)
            elif args.feature_type == 'Concat':
                concat_features = torch.cat((UNI_features, CHIEF_features, GigaPath_features), dim=1)
                output = model(concat_features)
            
            loss = criterion(output, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', lr, epoch)
        train_results = evaluate(model, criterion, train_loader, device, genes, args)  
        writer.add_scalar('Train Loss', train_results['loss'], epoch)
        writer.add_scalar('Train Mean Pearson', train_results['pearson_mean'], epoch)
        writer.add_scalar('Train Median L2 error', train_results['l2_error_q2'], epoch)
        writer.add_scalar('Train Median r2 scores', train_results['r2_score_q2'], epoch)            
            
    return model

def test(model, test_loader, genes, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    with torch.no_grad():
        pred_gather, targets_gather = None, None
        for _, batch in enumerate(test_loader):
            UNI_features, CHIEF_features, GigaPath_features, targets = batch
            UNI_features, CHIEF_features, GigaPath_features, targets = UNI_features.to(device), CHIEF_features.to(device), GigaPath_features.to(device), targets.to(device)

            if args.feature_type == 'UNI':
                output = model(UNI_features)
            elif args.feature_type == 'CHIEF':
                output = model(CHIEF_features)
            elif args.feature_type == 'GigaPath':
                output = model(GigaPath_features)
            elif args.feature_type == 'Concat':
                concat_features = torch.cat((UNI_features, CHIEF_features, GigaPath_features), dim=1)
                output = model(concat_features)
            
            if pred_gather is None:
                pred_gather = output.cpu().numpy()
                targets_gather = targets.cpu().numpy()
            else:
                pred_gather = np.concatenate((pred_gather, output.cpu().numpy()), axis=0)
                targets_gather = np.concatenate((targets_gather, targets.cpu().numpy()), axis=0)
    
    errors = []
    r2_scores = []
    pearson_corrs = []
    pearson_genes = []
    for target_gene in range(targets_gather.shape[1]):
        pred_value = pred_gather[:, target_gene]
        targets_value = targets_gather[:, target_gene]
        l2_error = float(np.mean((pred_value - targets_value)**2))
        r2_score = float(1 - np.sum((targets_value - pred_value)**2) / np.sum((targets_value - np.mean(targets_value))**2))
        pearson_corr, _ = pearsonr(targets_value, pred_value)
        if np.isnan(pearson_corr):
            print(targets_value)
            print(pred_value)
        errors.append(l2_error)
        r2_scores.append(r2_score)
        pearson_corrs.append(pearson_corr)
        score_dict = {
            'name': genes[target_gene],
            'pearson_corr': pearson_corr,
        }
        pearson_genes.append(score_dict)
    
    results = {
            'l2_errors': list(errors), 
            'r2_scores': list(r2_scores),
            'pearson_corrs': pearson_genes,
            'pearson_mean': float(np.mean(pearson_corrs)),
            'pearson_std': float(np.std(pearson_corrs)),
            'l2_error_q1': float(np.percentile(errors, 25)),
            'l2_error_q2': float(np.median(errors)),
            'l2_error_q3': float(np.percentile(errors, 75)),
            'r2_score_q1': float(np.percentile(r2_scores, 25)),
            'r2_score_q2': float(np.median(r2_scores)),
            'r2_score_q3': float(np.percentile(r2_scores, 75))}
    
    return results

def main():
    args = argparser.parse_args()
    print(args)
    
    if args.feature_type == 'UNI':
        feature_num = 1024
    elif args.feature_type == 'CHIEF':
        feature_num = 768
    elif args.feature_type == 'GigaPath':
        feature_num = 1536
    elif args.feature_type == 'Concat':
        feature_num = 3328
        
    adata_root = './dataset/HEST_format'
    features_root = './embedding_features/merged'
        
    patient_list = os.listdir(features_root)
    filtered_patient_list = [patient for patient in patient_list if patient.startswith("Her2ST")]

    filtered_patient_list.sort()
    
    test_results_dict = []
    
    for test_patient_id in filtered_patient_list:
        print('test_patient_id:', test_patient_id)
        
        train_patient_list = [patient for patient in filtered_patient_list if patient != test_patient_id]
        print("len(train_patient_list):", len(train_patient_list))
                
        predict_gene_path = os.path.join(adata_root, 'Her2ST_target_genes.txt')
        with open(predict_gene_path, "r", encoding="utf-8") as f:
            genes = f.read().splitlines()
        gene_num = len(genes)
        print('gene_num:', gene_num)
        
        split_assets = {}
        for patient_id in train_patient_list:            
            patient_adata_path = os.path.join(adata_root, 'Her2ST', patient_id.split('_')[1])
                
            patient_features_path = os.path.join(features_root, patient_id)
            tissue_list = os.listdir(patient_features_path)
            for tissue_name in tissue_list:
                tissue_id = tissue_name.split('.')[0]
                embed_path = os.path.join(patient_features_path, f'{tissue_id}.h5')
                expr_path = os.path.join(patient_adata_path, tissue_id, 'aligned_adata.h5ad')
                
                assets, _ = read_assets_from_h5(embed_path)
                barcodes = assets['barcodes'].flatten().astype(str).tolist()
                adata = load_adata(expr_path, genes=genes, barcodes=barcodes, normalize=True)
                assets['gene_expression'] = adata.values
                split_assets = merge_dict(split_assets, assets)
            
        for key, val in split_assets.items(): 
            split_assets[key] = np.concatenate(val, axis=0)
                
        train_UNI_features = torch.Tensor(split_assets['UNI_features'])
        train_CHIEF_features = torch.Tensor(split_assets['CHIEF_features'])
        train_GigaPath_features = torch.Tensor(split_assets['GigaPath_features'])
        train_gene_expression = torch.Tensor(split_assets['gene_expression'])
        
        train_dataset = HESTDataset(train_UNI_features, train_CHIEF_features, train_GigaPath_features, train_gene_expression)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            
        model = LinearProbeReg(feature_num, gene_num)
        output_dir = os.path.join(args.output_root, test_patient_id)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model = train_eval(model, train_loader, output_dir, genes, args)
        torch.save(model.state_dict(), os.path.join(output_dir, 'model.pth'))
            
        if test_patient_id.startswith('Her2ST'):
            test_adata_path = os.path.join(adata_root, 'Her2ST', test_patient_id.split('_')[1])
        else:
            test_adata_path = os.path.join(adata_root, 'in_situ', test_patient_id)
            
        test_features_path = os.path.join(features_root, test_patient_id)
        test_tissue_list = os.listdir(test_features_path)
        for test_tissue_name in test_tissue_list:
            test_tissue_id = test_tissue_name.split('.')[0]
            test_embed_path = os.path.join(test_features_path, f'{test_tissue_id}.h5')
            test_expr_path = os.path.join(test_adata_path, test_tissue_id, 'aligned_adata.h5ad')
            test_assets, _ = read_assets_from_h5(test_embed_path)
            test_barcodes = test_assets['barcodes'].flatten().astype(str).tolist()
            test_adata = load_adata(test_expr_path, genes=genes, barcodes=test_barcodes, normalize=True)
            test_assets['gene_expression'] = test_adata.values
        
            test_UNI_features = torch.Tensor(test_assets['UNI_features'])
            test_CHIEF_features = torch.Tensor(test_assets['CHIEF_features'])
            test_GigaPath_features = torch.Tensor(test_assets['GigaPath_features'])
            test_gene_expression = torch.Tensor(test_assets['gene_expression'])
        
            test_dataset = HESTDataset(test_UNI_features, test_CHIEF_features, test_GigaPath_features, test_gene_expression)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            
            test_results = test(model, test_loader, genes, args)

            pearson_corrs_df = pd.DataFrame(test_results['pearson_corrs'])
            pearson_csv_file = os.path.join(output_dir, f'pearson_{test_tissue_id}.csv')
            pearson_corrs_df.to_csv(pearson_csv_file, index=False)
        
            test_result_record = {'test_index': test_patient_id, 'tissue_index': test_tissue_id , 'pearson':test_results['pearson_mean']}
            test_results_dict.append(test_result_record)
            
    test_results_df = pd.DataFrame(test_results_dict)
    pearson_mean = test_results_df['pearson'].mean()
    pearson_std = test_results_df['pearson'].std()
    mean_row = pd.DataFrame({'pearson': [pearson_mean], 'test_index': ['mean']})
    std_row = pd.DataFrame({'pearson': [pearson_std], 'test_index': ['std']})
    test_results_df = pd.concat([test_results_df, mean_row, std_row], ignore_index=True)
    
    test_results_df.to_csv(os.path.join(args.output_root, 'test_results.csv'), index=False, float_format='%.4f')

# Make argparser
argparser = argparse.ArgumentParser(description='Linear Probe')
argparser.add_argument('--feature_type',        type=str, default='UNI', choices=['UNI', 'CHIEF', 'GigaPath', 'Concat'], help='Feature type')
argparser.add_argument('--batch_size',          type=int, default=512, help='Batch size')
argparser.add_argument('--epochs_num',          type=int, default=20, help='The number of epochs')
argparser.add_argument('--n_cycles',            type=int, default=1, help='Number of CosineAnnealingLR cycles')
argparser.add_argument('--lr',                  type=float, default=1e-4, help='Learning rate')
argparser.add_argument('--min_lr',              type=float, default=0.0, help='Minimum learning rate')
argparser.add_argument('--weight_decay',        type=float, default=0.0, help='Weight decay')
argparser.add_argument('--num_workers',         type=int, default=10, help='Number of workers')
argparser.add_argument('--output_root',         type=str, default='outputs', help='Output root path')

if __name__ == '__main__':
    main()