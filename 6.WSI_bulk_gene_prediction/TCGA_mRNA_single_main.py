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
import torch.nn.functional as F
import torch.nn as nn

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr

class CustomLoss(nn.Module):
    def __init__(self, coef_weight):
        super(CustomLoss, self).__init__()
        self.coef_weight = coef_weight

    def forward(self, y_pred, y_true):
        mse_loss = nn.MSELoss()(y_pred, y_true)
        y_pred_cpu = y_pred.detach().cpu().numpy()
        y_true_cpu = y_true.detach().cpu().numpy()
        
        correlation_loss = 1 - torch.mean(torch.tensor([np.corrcoef(y_pred_cpu[:, i], y_true_cpu[:, i])[0, 1] for i in range(y_pred.shape[1])]))
        loss = mse_loss + self.coef_weight * correlation_loss
        
        return loss, mse_loss, correlation_loss

class MutDataset(Dataset):
    def __init__(self, WSI_features, RNA_expression):
        self.WSI_features = WSI_features
        self.RNA_expression = RNA_expression

    def __len__(self):
        return len(self.WSI_features)

    def __getitem__(self, idx):
        WSI_features_item = self.WSI_features[idx]
        RNA_expression_item = self.RNA_expression[idx]
        
        return WSI_features_item, RNA_expression_item

class LinearProbeReg(torch.nn.Module):

    def __init__(self, embed_dim, latent_dim, gene_num):
        super(LinearProbeReg, self).__init__()

        self.linear = torch.nn.Linear(embed_dim, latent_dim)
        self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(latent_dim, gene_num)

    def forward(self, x):
        x = self.relu(self.linear(x))
        x = self.fc(x)
        
        return x

def evaluate(model, criterion, val_loader, device, target_genes, args):
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_mse_loss = 0
        total_coef_loss = 0

        pred_gather, targets_gather = None, None
        for _, batch in enumerate(val_loader):
            WSI_features, RNA_expression = batch
            WSI_features, RNA_expression = WSI_features.to(device), RNA_expression.to(device)
            
            output = model(WSI_features)
            loss, mse_loss, coef_loss = criterion(output, RNA_expression)
            
            total_loss += loss.item()
            total_mse_loss += mse_loss.item()
            total_coef_loss += coef_loss.item()
            
            if pred_gather is None:
                pred_gather = output.cpu().numpy()
                targets_gather = RNA_expression.cpu().numpy()
            else:
                pred_gather = np.concatenate((pred_gather, output.cpu().numpy()), axis=0)
                targets_gather = np.concatenate((targets_gather, RNA_expression.cpu().numpy()), axis=0)
    
    val_loss = total_loss / len(val_loader)
    val_mse_loss = total_mse_loss / len(val_loader)
    val_coef_loss = total_coef_loss / len(val_loader)
    
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
            print("pred_value:", pred_value)
            print("targets_value:", targets_value)
        errors.append(l2_error)
        r2_scores.append(r2_score)
        pearson_corrs.append(pearson_corr)
        score_dict = {
            'name': target_genes[target_gene],
            'pearson_corr': pearson_corr,
        }
        pearson_genes.append(score_dict)
    
    results = {'loss': val_loss,
            'mse_loss': val_mse_loss,
            'coef_loss': val_coef_loss,
            'l2_errors': list(errors), 
            'r2_scores': list(r2_scores),
            'pearson_corrs': pearson_genes,
            'pearson_mean': float(np.mean(pearson_corrs)),
            'pearson_std': float(np.std(pearson_corrs))}
    
    return results

def test(model, test_loader, test_case_id, device, target_genes, args):
    # Evaluate the model
    model.eval()
    with torch.no_grad():

        pred_gather, targets_gather = None, None
        for _, batch in enumerate(test_loader):
            WSI_features, RNA_expression = batch
            WSI_features, RNA_expression = WSI_features.to(device), RNA_expression.to(device)
            
            output = model(WSI_features)

            if pred_gather is None:
                pred_gather = output.cpu().numpy()
                targets_gather = RNA_expression.cpu().numpy()
            else:
                pred_gather = np.concatenate((pred_gather, output.cpu().numpy()), axis=0)
                targets_gather = np.concatenate((targets_gather, RNA_expression.cpu().numpy()), axis=0)
    
    pred_df = pd.DataFrame(
        data = pred_gather,
        index = test_case_id,
        columns = target_genes
    )

    targets_df = pd.DataFrame(
        data = targets_gather,
        index = test_case_id,
        columns = target_genes
    )

    agg_dict = {gene: 'mean' for gene in target_genes}
    
    pred_agg_df = pred_df.groupby(level=0).agg(agg_dict)
    targets_agg_df = targets_df.groupby(level=0).agg(agg_dict)
    
    errors = []
    r2_scores = []
    pearson_corrs = []
    pearson_genes = []
    for target_gene in range(targets_agg_df.shape[1]):
        pred_value = pred_agg_df.iloc[:, target_gene]
        targets_value = targets_agg_df.iloc[:, target_gene]
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
            'name': target_genes[target_gene],
            'pearson_corr': pearson_corr,
        }
        pearson_genes.append(score_dict)
    
    results = {'l2_errors': list(errors), 
            'r2_scores': list(r2_scores),
            'pearson_corrs': pearson_genes,
            'pearson_mean': float(np.mean(pearson_corrs)),
            'pearson_std': float(np.std(pearson_corrs))}
    
    return results
    
def train_eval(model,
          train_loader,
          test_loader,
          test_case_id,
          output_dir,
          target_genes,
          args):

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # set Tensorboard
    tensorboard_dir = os.path.join(output_dir, 'tensorboard')
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = tensorboard.SummaryWriter(tensorboard_dir)
    
    criterion = CustomLoss(args.coef_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Set the learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_num*len(train_loader) // args.n_cycles , eta_min=args.min_lr)
    
    for epoch in range(args.epochs_num):
        for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs_num}"):
            WSI_features, RNA_expression = batch
            WSI_features, RNA_expression = WSI_features.to(device), RNA_expression.to(device)
            
            output = model(WSI_features)
            
            loss, _, _ = criterion(output, RNA_expression)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', lr, epoch)
        train_results = evaluate(model, criterion, train_loader, device, target_genes, args)
        writer.add_scalar('Train Loss', train_results['loss'], epoch)
        writer.add_scalar('Train MES Loss', train_results['mse_loss'], epoch)
        writer.add_scalar('Train Coef Loss', train_results['coef_loss'], epoch)
        writer.add_scalar('Train Pearson Correlation', train_results['pearson_mean'], epoch)         

    test_results = test(model, test_loader, test_case_id, device, target_genes, args)
    writer.add_scalar(f'Test Pearson Correlation', test_results['pearson_mean'])
    
    return test_results

def main():
    args = argparser.parse_args()
    print(args)
    
    WSI_features_root = './embedding_features'
    expression_root = './target_genes_expression'
    split_root = './splits_five_fold'
    
    expression_df = pd.read_csv(os.path.join(expression_root, f'{args.cancer_type}_mRNA_top50_log2.csv'))
    WSI_features_path = os.path.join(WSI_features_root, args.cancer_type)
    target_genes = expression_df.columns[3:].tolist()
    print("len(target_genes):", len(target_genes))

    WSI_features_type_path = os.path.join(WSI_features_path, f'FEATURES_DIRECTORY_{args.feature_type}_WSI_features')
    
    if args.feature_type == 'TITAN':
        feature_num = 768
    elif args.feature_type == 'PRISM':
        feature_num = 1280
        
    fold_list = ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']
    
    test_results_list = []
    
    for version_index in range(10):
        with open(os.path.join(split_root, f'{args.cancer_type}_FF_splits_{version_index}.json'), 'r') as f:
            folds_dict = json.load(f)
        for fold_name in fold_list:
            print('fold_name:', fold_name)
            
            train_ids = folds_dict[fold_name]['train_case_ids']
            test_ids = folds_dict[fold_name]['test_case_ids']
            
            expression_train = expression_df[expression_df['TCGA_case_id'].isin(train_ids)]
            expression_test = expression_df[expression_df['TCGA_case_id'].isin(test_ids)]
            test_case_id = expression_test['TCGA_case_id'].values
            
            train_tensors_list = []
            for slide_name in expression_train['HE_entity_submitter_id']:
                pt_file_path = os.path.join(WSI_features_type_path, slide_name+'.pt')
                pt_data = torch.load(pt_file_path)
                
                if isinstance(pt_data, np.ndarray):
                    pt_data = torch.tensor(pt_data)
                elif isinstance(pt_data, torch.Tensor):
                    pt_data = pt_data
                else:
                    raise TypeError("Unsupported data type")
                        
                train_tensors_list.append(pt_data)
                
            train_tensors_concat = torch.cat(train_tensors_list, dim=0)
            train_labels_np = expression_train[target_genes].values
            train_labels_tensor = torch.tensor(train_labels_np, dtype=torch.float32)
            
            test_tensors_list = []
            for slide_name in expression_test['HE_entity_submitter_id']:

                pt_file_path = os.path.join(WSI_features_type_path, slide_name+'.pt')
                pt_data = torch.load(pt_file_path)
                
                if isinstance(pt_data, np.ndarray):
                    pt_data = torch.tensor(pt_data)
                elif isinstance(pt_data, torch.Tensor):
                    pt_data = pt_data
                else:
                    raise TypeError("Unsupported data type")
                
                test_tensors_list.append(pt_data)
                
            test_tensors_concat = torch.cat(test_tensors_list, dim=0)
            test_labels_np = expression_test[target_genes].values
            test_labels_tensor = torch.tensor(test_labels_np, dtype=torch.float32)

            train_dataset = MutDataset(train_tensors_concat, train_labels_tensor)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
            
            test_dataset = MutDataset(test_tensors_concat, test_labels_tensor)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
                        
            model = LinearProbeReg(feature_num, args.latent_dim, len(target_genes))
            output_dir = os.path.join(args.output_root, args.cancer_type, args.version, f"{version_index}_{fold_name}")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            test_results = train_eval(model, train_loader, test_loader, test_case_id, output_dir, target_genes, args)
            
            pearson_corrs_df = pd.DataFrame(test_results['pearson_corrs'])
            pearson_csv_file = os.path.join(output_dir, 'pearson_results.csv')
            pearson_corrs_df.to_csv(pearson_csv_file, index=False)
            
            test_result_record = {'version_index': version_index, 'fold_name': fold_name, 'pearson':test_results['pearson_mean']}
            test_results_list.append(test_result_record)
            
    output_path = os.path.join(args.output_root, args.cancer_type, args.version)
    test_results_df = pd.DataFrame(test_results_list)
    test_results_df.to_csv(os.path.join(output_path, 'test_results.csv'), index=False, float_format='%.4f')
    
    pearson_mean = test_results_df['pearson'].mean()
    pearson_std = test_results_df['pearson'].std()
    average_row = pd.DataFrame([['average', 'average', pearson_mean], ['std', 'std', pearson_std]], columns=['version_index', 'fold_name', 'pearson'])
    summary_test_df = pd.concat([test_results_df, average_row], ignore_index=True)
    
    summary_test_df.to_csv(os.path.join(output_path, 'test_results_summary.csv'), index=False, float_format='%.4f')
    
# Make argparser
argparser = argparse.ArgumentParser(description='Linear Probe')
argparser.add_argument('--feature_type',        type=str, default='TITAN', choices=['TITAN', 'PRISM'], help='Feature type')
argparser.add_argument('--latent_dim',          type=int, default=512, help='dimension of latent space')
argparser.add_argument('--cancer_type',         type=str, default='BRCA', choices=['BRCA', 'CRC', 'NSCLC'], help='Cancer type')
argparser.add_argument('--batch_size',          type=int, default=512, help='Batch size')
argparser.add_argument('--epochs_num',          type=int, default=20, help='The number of epochs')
argparser.add_argument('--n_cycles',            type=int, default=1, help='Number of CosineAnnealingLR cycles')
argparser.add_argument('--lr',                  type=float, default=1e-4, help='Learning rate')
argparser.add_argument('--min_lr',              type=float, default=0.0, help='Minimum learning rate')
argparser.add_argument('--weight_decay',        type=float, default=0.0, help='Weight decay')
argparser.add_argument('--num_workers',         type=int, default=10, help='Number of workers')
argparser.add_argument('--output_root',         type=str, default='results', help='Output root path')
argparser.add_argument('--version',             type=str, default='v1', help='Model training version')
argparser.add_argument('--coef_weight',         type=float, default=1.0, help='Coef loss weight')


if __name__ == '__main__':
    main()