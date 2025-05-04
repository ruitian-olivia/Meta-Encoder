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
from attention_utils import MultiheadAttention

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

    def __init__(self, embed_dim, num_heads, encoding_size, gene_num):
        super(LinearProbeReg, self).__init__()
        
        self.cross_attention = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, kdim=encoding_size, vdim=encoding_size, qdim=encoding_size)

        self.fc = torch.nn.Linear(embed_dim, gene_num)

    def forward(self, x):
        x = x.unsqueeze(0)
        x, _ = self.cross_attention(x, x, x)
        x = x.squeeze()
        
        output = self.fc(x)
        
        return output

def evaluate(model, criterion, val_loader, device, genes, args):
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        total_loss = 0

        pred_gather, targets_gather = None, None
        for _, batch in enumerate(val_loader):
            UNI_features, CHIEF_features, GigaPath_features, targets = batch
            UNI_features, CHIEF_features, GigaPath_features, targets = UNI_features.to(device), CHIEF_features.to(device), GigaPath_features.to(device), targets.to(device)

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
            
            concat_features = torch.cat((UNI_features, CHIEF_features, GigaPath_features), dim=1)
            
            output = model(concat_features)
            
            if output.dim() == 1:
                output = output.unsqueeze(0)
            
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
    
    task_list = ['CCRCC',
             'COAD',
             'IDC',
             'LUNG',
             'LYMPH_IDC',
             'PAAD',
             'PRAD',
             'READ',
             'SKCM']
    
    test_results_dict = []
    
    for task_name in task_list:
        print('task_name:', task_name)
        
        bench_data_root = f'./hest-bench/{task_name}'
        features_root = f'./embedding_features/merged/{task_name}'
        splits_root = f'./hest-bench/{task_name}/splits'
        
        with open(os.path.join(bench_data_root, 'var_50genes.json'), 'r') as f:
            genes = json.load(f)['genes']
            gene_num = len(genes)
            print('gene_num:', gene_num)
        
        train_files = glob.glob(os.path.join(splits_root, 'train_*.csv'))
        test_files = glob.glob(os.path.join(splits_root, 'test_*.csv'))
        
        for train_file in train_files:
            train_index = train_file.split('_')[-1].split('.')[0]
            print("Train / test index:", train_index)
            test_file = os.path.join(splits_root, f'test_{train_index}.csv')
            
            try:
                if test_file not in test_files:
                    raise FileNotFoundError(f"Missing test file for {train_file}")
                
                print(f"Found matching pair: {train_file} and {test_file}")
            except FileNotFoundError as e:
                print(e)
                
            train_df = pd.read_csv(train_file)
            test_df = pd.read_csv(test_file)
            
            split_assets = {}
            for i in tqdm(range(len(train_df))):
                sample_id = train_df.iloc[i]['sample_id']
                embed_path = os.path.join(features_root, f'{sample_id}.h5')
                expr_path = os.path.join(bench_data_root, train_df.iloc[i]['expr_path'])
                
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
            
            model = LinearProbeReg(args.embed_dim, args.num_heads, args.feature_num, gene_num)
            output_dir = os.path.join(args.output_root, task_name, "fold_"+str(train_index))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model = train_eval(model, train_loader, output_dir, genes, args)
            torch.save(model.state_dict(), os.path.join(output_dir, 'model.pth'))
            
            for test_i in tqdm(range(len(test_df))):
                test_sample_id = test_df.iloc[test_i]['sample_id']
                embed_path = os.path.join(features_root, f'{test_sample_id}.h5')
                expr_path = os.path.join(bench_data_root, test_df.iloc[test_i]['expr_path'])
                
                test_assets, _ = read_assets_from_h5(embed_path)
                barcodes = test_assets['barcodes'].flatten().astype(str).tolist()
                adata = load_adata(expr_path, genes=genes, barcodes=barcodes, normalize=True)
                test_assets['gene_expression'] = adata.values
            
                test_UNI_features = torch.Tensor(test_assets['UNI_features'])
                test_CHIEF_features = torch.Tensor(test_assets['CHIEF_features'])
                test_GigaPath_features = torch.Tensor(test_assets['GigaPath_features'])
                test_gene_expression = torch.Tensor(test_assets['gene_expression'])
            
                test_dataset = HESTDataset(test_UNI_features, test_CHIEF_features, test_GigaPath_features, test_gene_expression)
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
                
                test_results = test(model, test_loader, genes, args)

                pearson_corrs_df = pd.DataFrame(test_results['pearson_corrs'])
                pearson_csv_file = os.path.join(output_dir, f'pearson_{test_sample_id}.csv')
                pearson_corrs_df.to_csv(pearson_csv_file, index=False)
            
                test_result_record = {'task_name': task_name, 'train_index': train_index, 'tissue_index': test_sample_id , 'pearson':test_results['pearson_mean']}
                test_results_dict.append(test_result_record)
            
    test_results_df = pd.DataFrame(test_results_dict)
    test_results_df.to_csv(os.path.join(args.output_root, 'test_results.csv'), index=False, float_format='%.4f')
    
    summary_test_df = test_results_df.groupby('task_name')['pearson'].agg([np.mean, np.std]).reset_index()
    summary_test_df.columns = ['task_name', 'pearson_mean', 'pearson_std']
    
    pearson_mean_avg = summary_test_df['pearson_mean'].mean()
    pearson_std_avg = summary_test_df['pearson_std'].mean()
    average_row = pd.DataFrame([['average', pearson_mean_avg, pearson_std_avg]], columns=['task_name', 'pearson_mean', 'pearson_std'])
    summary_test_df = pd.concat([summary_test_df, average_row], ignore_index=True)
    
    summary_test_df.to_csv(os.path.join(args.output_root, 'test_results_summary.csv'), index=False, float_format='%.4f')
  
# Make argparser
argparser = argparse.ArgumentParser(description='Linear Probe')
argparser.add_argument('--feature_num',         type=int, default=3328, help='Feature number')
argparser.add_argument('--embed_dim',           type=int, default=1024, help='Embedding dimension')
argparser.add_argument('--num_heads',           type=int, default=1, help='The number of attention heads')
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