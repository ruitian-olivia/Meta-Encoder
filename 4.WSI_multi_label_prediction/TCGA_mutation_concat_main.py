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

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt) ** self.gamma * BCE_loss

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            F_loss = alpha_t * F_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

class MutDataset(Dataset):
    def __init__(self, WSI_features, mutation_labels):
        self.WSI_features = WSI_features
        self.mutation_labels = mutation_labels

    def __len__(self):
        return len(self.WSI_features)

    def __getitem__(self, idx):
        WSI_features_item = self.WSI_features[idx]
        mutation_labels_item = self.mutation_labels[idx]
        
        return WSI_features_item, mutation_labels_item

class LinearProbeReg(torch.nn.Module):

    def __init__(self, embed_dim: int = 512, gene_num: int = 3):
        super(LinearProbeReg, self).__init__()

        self.fc = torch.nn.Linear(embed_dim, gene_num)

    def forward(self, x):
        x = torch.sigmoid(self.fc(x))
        return x

def evaluate(model, criterion_list, val_loader, device, args):
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        total_loss = 0

        pred_gather, targets_gather = None, None
        for _, batch in enumerate(val_loader):
            WSI_features, mutation_labels = batch
            WSI_features, mutation_labels = WSI_features.to(device), mutation_labels.to(device)
            
            output = model(WSI_features)
            # loss = criterion(output, mutation_labels)

            loss1 = criterion_list[0](output[:, 0], mutation_labels[:, 0])
            loss2 = criterion_list[1](output[:, 1], mutation_labels[:, 1])
            loss3 = criterion_list[2](output[:, 2], mutation_labels[:, 2])
            loss = (loss1 + loss2 + loss3) / 3
            
            total_loss += loss.item()
            
            if pred_gather is None:
                pred_gather = output.cpu().numpy()
                targets_gather = mutation_labels.cpu().numpy()
            else:
                pred_gather = np.concatenate((pred_gather, output.cpu().numpy()), axis=0)
                targets_gather = np.concatenate((targets_gather, mutation_labels.cpu().numpy()), axis=0)
    
    val_loss = total_loss / len(val_loader)
    
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    roc_aucs = []
    
    for target_gene in range(targets_gather.shape[1]):
        pred_value = pred_gather[:, target_gene]
        pred_binary = (pred_value >= 0.5).astype(int)
        
        targets_value = targets_gather[:, target_gene]

        accuracy = accuracy_score(targets_value, pred_binary)
        precision = precision_score(targets_value, pred_binary)
        recall = recall_score(targets_value, pred_binary)
        f1 = f1_score(targets_value, pred_binary)
        roc_auc = roc_auc_score(targets_value, pred_value)
        
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        roc_aucs.append(roc_auc)
    
    return val_loss, accuracies, precisions, recalls, f1_scores, roc_aucs

def test(model, test_loader, test_case_id, device, args):
    # Evaluate the model
    model.eval()
    with torch.no_grad():

        pred_gather, targets_gather = None, None
        for _, batch in enumerate(test_loader):
            WSI_features, mutation_labels = batch
            WSI_features, mutation_labels = WSI_features.to(device), mutation_labels.to(device)
            
            output = model(WSI_features)

            if pred_gather is None:
                pred_gather = output.cpu().numpy()
                targets_gather = mutation_labels.cpu().numpy()
            else:
                pred_gather = np.concatenate((pred_gather, output.cpu().numpy()), axis=0)
                targets_gather = np.concatenate((targets_gather, mutation_labels.cpu().numpy()), axis=0)
                
    test_results = {
        'test_case_id': test_case_id
    }
    
    for target_gene in range(targets_gather.shape[1]):
        pred_value = pred_gather[:, target_gene]
        target_value = targets_gather[:, target_gene]
        
        test_results[f'pred_value_{target_gene}'] = pred_value
        test_results[f'target_value_{target_gene}'] = target_value
        
    test_results_df = pd.DataFrame(test_results)
    
    pred_cols = test_results_df.filter(like='pred_value_').columns
    targets_cols = test_results_df.filter(like='target_value_').columns

    agg_dict = {col: 'max' for col in pred_cols}
    agg_dict.update({col: 'mean' for col in targets_cols})
    
    test_results_df = test_results_df.groupby('test_case_id').agg(agg_dict).reset_index()
    
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    roc_aucs = []
    
    for target_gene in range(targets_gather.shape[1]):
        pred_value = test_results_df[f'pred_value_{target_gene}']
        pred_binary = (pred_value >= 0.5).astype(int)
        
        targets_value = test_results_df[f'target_value_{target_gene}']

        accuracy = accuracy_score(targets_value, pred_binary)
        precision = precision_score(targets_value, pred_binary)
        recall = recall_score(targets_value, pred_binary)
        f1 = f1_score(targets_value, pred_binary)
        roc_auc = roc_auc_score(targets_value, pred_value)
        
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        roc_aucs.append(roc_auc)
    
    return accuracies, precisions, recalls, f1_scores, roc_aucs
    
def train_eval(model,
          train_loader,
          test_loader,
          test_case_id,
          output_dir,
          args):

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # set Tensorboard
    tensorboard_dir = os.path.join(output_dir, 'tensorboard')
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = tensorboard.SummaryWriter(tensorboard_dir)
    
    # criterion = torch.nn.BCEWithLogitsLoss()
    criterion1 = FocalLoss(alpha=args.alpha_values[0], gamma=2)
    criterion2 = FocalLoss(alpha=args.alpha_values[1], gamma=2)
    criterion3 = FocalLoss(alpha=args.alpha_values[2], gamma=2)
    criterion_list = [criterion1, criterion2, criterion3]

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Set the learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_num*len(train_loader) // args.n_cycles , eta_min=args.min_lr)
    
    for epoch in range(args.epochs_num):
        for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs_num}"):
            WSI_features, mutation_labels = batch
            WSI_features, mutation_labels = WSI_features.to(device), mutation_labels.to(device)
            
            output = model(WSI_features)
            
            # loss = criterion(output, mutation_labels)
            loss1 = criterion_list[0](output[:, 0], mutation_labels[:, 0])
            loss2 = criterion_list[1](output[:, 1], mutation_labels[:, 1])
            loss3 = criterion_list[2](output[:, 2], mutation_labels[:, 2])
            
            loss = (loss1 + loss2 + loss3) / 3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', lr, epoch)
        train_loss, accuracies, precisions, recalls, f1_scores, roc_aucs = evaluate(model, criterion_list, train_loader, device, args)
        writer.add_scalar('Train Loss', train_loss, epoch)
        for index, gene in enumerate(args.predict_genes):
            # print(f"Index: {index}, Gene: {gene}")
         
            writer.add_scalar(f'Train {gene} Acc', accuracies[index], epoch)
            writer.add_scalar(f'Train {gene} precision', precisions[index], epoch)
            writer.add_scalar(f'Train {gene} recall', recalls[index], epoch)
            writer.add_scalar(f'Train {gene} f1_score', f1_scores[index], epoch)
            writer.add_scalar(f'Train {gene} roc_auc', roc_aucs[index], epoch)            
          
    test_accuracies, test_precisions, test_recalls, test_f1_scores, test_roc_aucs = test(model, test_loader,test_case_id, device, args)
    for index, gene in enumerate(args.predict_genes):
        
        writer.add_scalar(f'Test {gene} Acc', test_accuracies[index])
        writer.add_scalar(f'Test {gene} precision', test_precisions[index])
        writer.add_scalar(f'Test {gene} recall', test_recalls[index])
        writer.add_scalar(f'Test {gene} f1_score', test_f1_scores[index])
        writer.add_scalar(f'Test {gene} roc_auc', test_roc_aucs[index])   
    test_results_dict = {
        'Gene name': args.predict_genes,
        'Test Accuracy': test_accuracies,
        'Test Precision': test_precisions,
        'Test Recall': test_recalls,
        'Test F1 Score': test_f1_scores,
        'Test ROC AUC': test_roc_aucs
    }
    
    test_results_df = pd.DataFrame(test_results_dict)
    test_results_df.set_index('Gene name', inplace=True)
    test_csv_file = os.path.join(output_dir, 'test_resuts.csv')
    test_results_df.to_csv(test_csv_file, index=True, float_format='%.4f')
    
    return test_results_df

def main():
    args = argparser.parse_args()
    print(args)
    
    WSI_features_root = './embedding_features'
    mutation_root = './biomarker_info'
    split_root = './splits_five_fold'
    
    if args.cancer_type == 'BRCA':
        mutation_df = pd.read_csv(os.path.join(mutation_root, 'tcga_brca_mutation_info.csv'))
        WSI_features_path = os.path.join(WSI_features_root, 'BRCA')
        selected_mutation_columns = mutation_df.columns[:3].tolist() + args.predict_genes
        selected_mutation_df = mutation_df[selected_mutation_columns]
        
    elif args.cancer_type == 'CRC':
        mutation_df = pd.read_csv(os.path.join(mutation_root, 'CRC_MSI_mutation_info.csv'))
        WSI_features_path = os.path.join(WSI_features_root, 'CRC')
        selected_mutation_columns = ['TCGA_case_id','slide_id','slide_name'] + args.predict_genes
        selected_mutation_df = mutation_df[selected_mutation_columns]
        selected_mutation_df = selected_mutation_df.rename(columns={'TCGA_case_id': 'case_id'})

    
    if selected_mutation_df.isnull().values.any():
        print("selected_mutation_df contains missing values.")
    else:
        print("selected_mutation_df does not contain any missing values.")
        
    fold_list = ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']
    
    test_results_list = []
    
    for version_index in range(10):
        with open(os.path.join(split_root, f'{args.cancer_type}_FF_splits_{version_index}.json'), 'r') as f:
            folds_dict = json.load(f)
        for fold_name in fold_list:
            print('fold_name:', fold_name)
            
            train_ids = folds_dict[fold_name]['train_case_ids']
            test_ids = folds_dict[fold_name]['test_case_ids']
            
            mutation_train = selected_mutation_df[selected_mutation_df['case_id'].isin(train_ids)]
            mutation_test = selected_mutation_df[selected_mutation_df['case_id'].isin(test_ids)]
            test_case_id = mutation_test['case_id'].values
            
            train_tensors_list = []
            for slide_name in mutation_train['slide_name']:
                features_tensor_list = []
                for feature_type in args.feature_types:
                    WSI_features_type_path = os.path.join(WSI_features_path, f'FEATURES_DIRECTORY_{feature_type}_WSI_features')

                    pt_file_path = os.path.join(WSI_features_type_path, slide_name+'.pt')
                    pt_data = torch.load(pt_file_path)
                    if isinstance(pt_data, np.ndarray):
                        pt_data = torch.tensor(pt_data)
                    elif isinstance(pt_data, torch.Tensor):
                        pt_data = pt_data
                    else:
                        raise TypeError("Unsupported data type")

                    features_tensor_list.append(pt_data)
                    
                features_tensor_list = [tensor.to('cpu') for tensor in features_tensor_list]
                pt_data_concat = torch.cat(features_tensor_list, dim=1)
                train_tensors_list.append(pt_data_concat)
                
            train_tensors_concat = torch.cat(train_tensors_list, dim=0)
            train_labels_np = mutation_train[args.predict_genes].applymap(lambda x: 1 if x == 'MUT' else 0).values
            train_labels_tensor = torch.tensor(train_labels_np, dtype=torch.float32)
            
            test_tensors_list = []
            for slide_name in mutation_test['slide_name']:
                features_tensor_list = []
                for feature_type in args.feature_types:
                    WSI_features_type_path = os.path.join(WSI_features_path, f'FEATURES_DIRECTORY_{feature_type}_WSI_features')

                    pt_file_path = os.path.join(WSI_features_type_path, slide_name+'.pt')
                    pt_data = torch.load(pt_file_path)
                    if isinstance(pt_data, np.ndarray):
                        pt_data = torch.tensor(pt_data)
                    elif isinstance(pt_data, torch.Tensor):
                        pt_data = pt_data
                    else:
                        raise TypeError("Unsupported data type")

                    features_tensor_list.append(pt_data)
                    
                features_tensor_list = [tensor.to('cpu') for tensor in features_tensor_list]
                pt_data_concat = torch.cat(features_tensor_list, dim=1)
                test_tensors_list.append(pt_data_concat)
                
            test_tensors_concat = torch.cat(test_tensors_list, dim=0)
            test_labels_np = mutation_test[args.predict_genes].applymap(lambda x: 1 if x == 'MUT' else 0).values
            test_labels_tensor = torch.tensor(test_labels_np, dtype=torch.float32)

            train_dataset = MutDataset(train_tensors_concat, train_labels_tensor)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
            
            test_dataset = MutDataset(test_tensors_concat, test_labels_tensor)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
                        
            model = LinearProbeReg(args.feature_num, len(args.predict_genes))
            output_dir = os.path.join(args.output_root, args.cancer_type, args.version, f"{version_index}_{fold_name}")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            test_performance_df = train_eval(model, train_loader, test_loader, test_case_id, output_dir, args)
            
            test_results_list.append(test_performance_df)
        
    test_results_data = np.array([df.values for df in test_results_list])

    test_mean_df = pd.DataFrame(np.mean(test_results_data, axis=0), columns=test_results_list[0].columns, index=test_results_list[0].index)
    test_mean_df.to_csv(os.path.join(args.output_root, args.cancer_type, args.version, 'test_results_mean.csv'), float_format='%.4f')

    test_std_df = pd.DataFrame(np.std(test_results_data, axis=0), columns=test_results_list[0].columns, index=test_results_list[0].index)
    test_std_df.to_csv(os.path.join(args.output_root, args.cancer_type, args.version, 'test_results_std.csv'), float_format='%.4f')
    
# Make argparser
argparser = argparse.ArgumentParser(description='Linear Probe')
argparser.add_argument('--feature_types',       type=str, nargs='+', help='List of concated feature types')
argparser.add_argument('--feature_num',         type=int, default=2048, help='dimension of embedded features')
argparser.add_argument('--cancer_type',         type=str, default='BRCA', choices=['BRCA', 'LUAD', 'CRC'], help='Cancer type')
argparser.add_argument('--batch_size',          type=int, default=512, help='Batch size')
argparser.add_argument('--epochs_num',          type=int, default=20, help='The number of epochs')
argparser.add_argument('--n_cycles',            type=int, default=1, help='Number of CosineAnnealingLR cycles')
argparser.add_argument('--lr',                  type=float, default=1e-4, help='Learning rate')
argparser.add_argument('--min_lr',              type=float, default=0.0, help='Minimum learning rate')
argparser.add_argument('--weight_decay',        type=float, default=0.0, help='Weight decay')
argparser.add_argument('--num_workers',         type=int, default=10, help='Number of workers')
argparser.add_argument('--output_root',         type=str, default='results', help='Output root path')
argparser.add_argument('--version',             type=str, default='v1', help='Model training version')
argparser.add_argument('--predict_genes',       type=str, nargs='+', help='List of predicted genes')
argparser.add_argument('--alpha_values',        type=float, nargs='+', help='List of alpha values for Focal Loss')

if __name__ == '__main__':
    main()