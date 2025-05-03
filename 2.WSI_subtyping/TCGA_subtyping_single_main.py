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

def evaluate(model, criterion, val_loader, device, args):
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        total_loss = 0

        pred_gather, targets_gather = None, None
        for _, batch in enumerate(val_loader):
            WSI_features, mutation_labels = batch
            WSI_features, mutation_labels = WSI_features.to(device), mutation_labels.to(device)
            mutation_labels = mutation_labels.view(-1, 1)
            output = model(WSI_features)
            loss = criterion(output, mutation_labels)
            
            total_loss += loss.item()
            
            if pred_gather is None:
                pred_gather = output.cpu().numpy()
                targets_gather = mutation_labels.cpu().numpy()
            else:
                pred_gather = np.concatenate((pred_gather, output.cpu().numpy()), axis=0)
                targets_gather = np.concatenate((targets_gather, mutation_labels.cpu().numpy()), axis=0)
    
    val_loss = total_loss / len(val_loader)
    
    pred_binary = (pred_gather >= 0.5).astype(int)
    
    accuracy = accuracy_score(targets_gather, pred_binary)
    precision = precision_score(targets_gather, pred_binary)
    recall = recall_score(targets_gather, pred_binary)
    f1 = f1_score(targets_gather, pred_binary)
    roc_auc = roc_auc_score(targets_gather, pred_gather)
    
    return val_loss, accuracy, precision, recall, f1, roc_auc

def test(model, criterion, test_loader, test_case_id, device, args):
    model.eval()
    with torch.no_grad():
        total_loss = 0

        pred_gather, targets_gather = None, None
        for _, batch in enumerate(test_loader):
            WSI_features, mutation_labels = batch
            WSI_features, mutation_labels = WSI_features.to(device), mutation_labels.to(device)
            mutation_labels = mutation_labels.view(-1, 1)
            output = model(WSI_features)
            loss = criterion(output, mutation_labels)
            
            total_loss += loss.item()
            
            if pred_gather is None:
                pred_gather = output.cpu().numpy()
                targets_gather = mutation_labels.cpu().numpy()
            else:
                pred_gather = np.concatenate((pred_gather, output.cpu().numpy()), axis=0)
                targets_gather = np.concatenate((targets_gather, mutation_labels.cpu().numpy()), axis=0)
        
    test_results = {
        'test_case_id': test_case_id,
        'pred_gather': pred_gather.squeeze(),
        'targets_gather': targets_gather.squeeze()
    }

    test_results_df = pd.DataFrame(test_results)
    test_results_df = test_results_df.groupby('test_case_id').agg({
        'pred_gather': 'max',
        'targets_gather': 'mean'
    }).reset_index()
    
    pred_gather = test_results_df['pred_gather'].values
    targets_gather = test_results_df['targets_gather'].values
    pred_binary = (pred_gather >= 0.5).astype(int)
    
    accuracy = accuracy_score(targets_gather, pred_binary)
    precision = precision_score(targets_gather, pred_binary)
    recall = recall_score(targets_gather, pred_binary)
    f1 = f1_score(targets_gather, pred_binary)
    roc_auc = roc_auc_score(targets_gather, pred_gather)
    
    return accuracy, precision, recall, f1, roc_auc

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
    
    criterion = FocalLoss(alpha=args.alpha_values, gamma=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Set the learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_num*len(train_loader) // args.n_cycles , eta_min=args.min_lr)
    
    for epoch in range(args.epochs_num):
        for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs_num}"):
            WSI_features, mutation_labels = batch
            WSI_features, mutation_labels = WSI_features.to(device), mutation_labels.to(device)
            output = model(WSI_features)
            mutation_labels = mutation_labels.view(-1, 1)
            loss = criterion(output, mutation_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', lr, epoch)
        train_loss, train_accuracy, train_precision, train_recall, train_f1, train_roc_auc = evaluate(model, criterion, train_loader, device, args)
        writer.add_scalar('Train Loss', train_loss, epoch)
         
        writer.add_scalar(f'Train Acc', train_accuracy, epoch)
        writer.add_scalar(f'Train precision', train_precision, epoch)
        writer.add_scalar(f'Train recall', train_recall, epoch)
        writer.add_scalar(f'Train f1_score', train_f1, epoch)
        writer.add_scalar(f'Train roc_auc', train_roc_auc, epoch)            
          
    test_accuracy, test_precision, test_recall, test_f1, test_roc_auc = test(model, criterion, test_loader, test_case_id, device, args)        
    writer.add_scalar(f'Test Acc', test_accuracy)
    writer.add_scalar(f'Test precision', test_precision)
    writer.add_scalar(f'Test recall', test_recall)
    writer.add_scalar(f'Test f1_score', test_f1)
    writer.add_scalar(f'Test roc_auc', test_roc_auc)
    
    test_results_dict = {
        'Test Accuracy': test_accuracy,
        'Test Precision': test_precision,
        'Test Recall': test_recall,
        'Test F1 Score': test_f1,
        'Test ROC AUC': test_roc_auc
    }
    
    test_results_df = pd.DataFrame(test_results_dict, index=[0])
    test_csv_file = os.path.join(output_dir, 'test_resuts.csv')
    test_results_df.to_csv(test_csv_file, index=True, float_format='%.4f')
    
    return test_results_df

def main():
    args = argparser.parse_args()
    print(args)
    
    WSI_features_root = './embedding_features/WSI_level'
    subtyping_root = '../1.CLAM_patch_subtyping/CLAM_feature/subtype_list'
    split_root = './subtyping_label'
    
    if args.cancer_type == 'BRCA':
        subtyping_df = pd.read_csv(os.path.join(subtyping_root, 'tcga_brca_subset.csv'))
        WSI_features_path = os.path.join(WSI_features_root, 'BRCA')

    elif args.cancer_type == 'NSCLC':
        subtyping_df = pd.read_csv(os.path.join(subtyping_root, 'tcga_lung_subset.csv'))
        WSI_features_path = os.path.join(WSI_features_root, 'NSCLC')
    
    WSI_features_type_path = os.path.join(WSI_features_path, f'FEATURES_DIRECTORY_{args.feature_type}_WSI_features')
    
    if args.feature_type == 'TITAN':
        feature_num = 768
    elif args.feature_type == 'PRISM':
        feature_num = 1280

    fold_list = ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']
    
    test_results_list = []
    
    for version_index in range(10):
        with open(os.path.join(split_root, f'{args.cancer_type}_subtyping_FF_splits_{version_index}.json'), 'r') as f:
            folds_dict = json.load(f)
        for fold_name in fold_list:
            print(f'version_index({version_index}) & fold_name({fold_name})')
            
            train_ids = folds_dict[fold_name]['train_case_ids']
            test_ids = folds_dict[fold_name]['test_case_ids']
            
            subtyping_train = subtyping_df[subtyping_df['case_id'].isin(train_ids)]
            subtyping_test = subtyping_df[subtyping_df['case_id'].isin(test_ids)]
            test_case_id = subtyping_test['case_id'].values
            
            train_tensors_list = []
            for slide_name in subtyping_train['slide_name']:
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
            if args.cancer_type == 'BRCA':
                train_labels_np = subtyping_train['label'].apply(lambda x: 1 if x == 'ILC' else 0).values
            elif args.cancer_type == 'NSCLC':
                train_labels_np = subtyping_train['label'].apply(lambda x: 1 if x == 'LUAD' else 0).values
            train_labels_tensor = torch.tensor(train_labels_np, dtype=torch.float32)
            
            test_tensors_list = []
            for slide_name in subtyping_test['slide_name']:
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
            if args.cancer_type == 'BRCA':
                test_labels_np = subtyping_test['label'].apply(lambda x: 1 if x == 'ILC' else 0).values
            elif args.cancer_type == 'NSCLC':
                test_labels_np = subtyping_test['label'].apply(lambda x: 1 if x == 'LUAD' else 0).values            
            
            test_labels_tensor = torch.tensor(test_labels_np, dtype=torch.float32)

            train_dataset = MutDataset(train_tensors_concat, train_labels_tensor)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
            
            test_dataset = MutDataset(test_tensors_concat, test_labels_tensor)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
                        
            model = LinearProbeReg(feature_num, 1)
            output_dir = os.path.join(args.output_root, args.cancer_type, args.feature_type+f'_{args.version}', f'{version_index}_{fold_name}')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            test_performance_df = train_eval(model, train_loader, test_loader, test_case_id, output_dir, args)
            
            test_results_list.append(test_performance_df)
        
    test_results_data = np.array([df.values for df in test_results_list])

    test_mean_df = pd.DataFrame(np.mean(test_results_data, axis=0), columns=test_results_list[0].columns, index=test_results_list[0].index)
    test_mean_df.to_csv(os.path.join(args.output_root, args.cancer_type, args.feature_type+f'_{args.version}', 'test_results_mean.csv'), float_format='%.4f')

    test_std_df = pd.DataFrame(np.std(test_results_data, axis=0), columns=test_results_list[0].columns, index=test_results_list[0].index)
    test_std_df.to_csv(os.path.join(args.output_root, args.cancer_type, args.feature_type+f'_{args.version}', 'test_results_std.csv'), float_format='%.4f')
    
# Make argparser
argparser = argparse.ArgumentParser(description='Linear Probe')
argparser.add_argument('--feature_type',        type=str, default='TITAN', choices=['TITAN', 'PRISM', 'Concat'], help='Feature type')
argparser.add_argument('--cancer_type',         type=str, default='BRCA', choices=['BRCA', 'NSCLC'], help='Cancer type')
argparser.add_argument('--batch_size',          type=int, default=512, help='Batch size')
argparser.add_argument('--epochs_num',          type=int, default=20, help='The number of epochs')
argparser.add_argument('--n_cycles',            type=int, default=1, help='Number of CosineAnnealingLR cycles')
argparser.add_argument('--lr',                  type=float, default=1e-4, help='Learning rate')
argparser.add_argument('--min_lr',              type=float, default=0.0, help='Minimum learning rate')
argparser.add_argument('--weight_decay',        type=float, default=0.0, help='Weight decay')
argparser.add_argument('--num_workers',         type=int, default=10, help='Number of workers')
argparser.add_argument('--output_root',         type=str, default='results', help='Output root path')
argparser.add_argument('--version',             type=str, default='v1', help='Model training version')
argparser.add_argument('--alpha_values',        type=float, default=0.5, help='alpha values for Focal Loss')

if __name__ == '__main__':
    main()