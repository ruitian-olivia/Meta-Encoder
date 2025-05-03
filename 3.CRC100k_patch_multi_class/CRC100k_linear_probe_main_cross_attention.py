import os
import torch
import pickle
import argparse
import itertools
import numpy as np
import pandas as pd
import torch.utils.tensorboard as tensorboard

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    cohen_kappa_score,
    classification_report,
    roc_auc_score, 
    average_precision_score, 
    f1_score, 
    precision_recall_fscore_support
)
from attention_utils import MultiheadAttention

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        return sample, label
    
class LinearProbe(torch.nn.Module):

    def __init__(self, embed_dim, num_heads, encoding_size, num_classes):
        super(LinearProbe, self).__init__()

        self.cross_attention = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, kdim=encoding_size, vdim=encoding_size, qdim=encoding_size)

        self.fc = torch.nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = x.unsqueeze(0)
        x, _ = self.cross_attention(x, x, x)
        x = x.squeeze()
        
        output = self.fc(x)
        
        return output

def to_onehot(labels, num_classes):
    onehot = np.zeros((labels.shape[0], num_classes))
    onehot[np.arange(labels.shape[0]), labels] = 1
    return onehot

def evaluate(model, criterion, val_loader, device):
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        total_loss = 0

        pred_gather, target_gather = [], []
        for _, (embed, target) in enumerate(val_loader):

            embed, target = embed.to(device), target.to(device)

            # forward pass
            output = model(embed)
            loss = criterion(output, target)
            total_loss += loss.item()
            # gather the predictions and targets
            pred_gather.append(output.cpu().numpy())
            target_gather.append(target.cpu().numpy())
            
    val_loss = total_loss / len(val_loader)
    
    pred_gather = np.concatenate(pred_gather)
    target_gather = np.concatenate(target_gather)

    pred_gather_label = pred_gather.argmax(1)
    pred_gather_onehot = to_onehot(pred_gather_label, pred_gather.shape[1])
    target_gather_onehot = to_onehot(target_gather, pred_gather.shape[1])
    
    acc = (pred_gather_label == target_gather).mean()
    f1 = f1_score(target_gather, pred_gather_label, average='weighted')
    precision, recall, _, _ = precision_recall_fscore_support(target_gather, pred_gather_label, average='macro')

    auroc = roc_auc_score(target_gather_onehot, pred_gather, average='macro')
    auprc = average_precision_score(target_gather_onehot, pred_gather, average='macro')
    
    bacc = balanced_accuracy_score(target_gather, pred_gather_label)
    kappa = cohen_kappa_score(target_gather, pred_gather_label, weights="quadratic")
    cls_rep = classification_report(target_gather, pred_gather_label, output_dict=True, zero_division=0)
    weighted_f1 = cls_rep["weighted avg"]["f1-score"]

    return val_loss, acc, f1, precision, recall, auroc, auprc, bacc, kappa, weighted_f1

def train_eval(model,
          train_loader,
          val_loader,
          test_loader,
          output_dir,
          args):

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # set Tensorboard
    tensorboard_dir = os.path.join(output_dir, 'tensorboard')
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = tensorboard.SummaryWriter(tensorboard_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Set the learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_iters // args.n_cycles , eta_min=args.min_lr)

    # Set the loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Set the infinite train loader
    infinite_train_loader = itertools.cycle(train_loader)

    best_f1 = 0
    # Train the model
    print('Start training')
    for i, (embed, target) in enumerate(infinite_train_loader):

        if i >= args.train_iters:
            break

        embed, target = embed.to(device), target.to(device)

        # Forward pass
        output = model(embed)
        loss = criterion(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if (i + 1) % 100 == 0:
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Train Loss', loss.item(), i)
            writer.add_scalar('Learning Rate', lr, i)
               
        if (i + 1) % args.eval_interval == 0 or (i + 1) == args.train_iters:
            _, train_acc, train_f1, train_precision, train_recall, train_auroc, train_auprc,\
            train_bacc, train_kappa, train_weighted_f1  = evaluate(model, criterion, val_loader, device)
            
            writer.add_scalar('Train Accuracy', train_acc, i)
            writer.add_scalar('Train Balance Accuracy', train_bacc, i)
            writer.add_scalar('Train F1-score', train_f1, i)
            writer.add_scalar('Train Weighted F1-score', train_weighted_f1, i)
            writer.add_scalar('Train Precision', train_precision, i)
            writer.add_scalar('Train Recall', train_recall, i)
            writer.add_scalar('Train AUROC', train_auroc, i)
            writer.add_scalar('Train AUPRC', train_auprc, i)
            writer.add_scalar('Train Kappa', train_kappa, i)

    # Save the model
    torch.save(model.state_dict(), f'{output_dir}/model.pth')

    # Evaluate the model
    _, test_acc, test_f1, test_precision, test_recall, test_auroc, test_auprc,\
    test_bacc, test_kappa, test_weighted_f1  = evaluate(model, criterion, test_loader, device)

    writer.add_scalar('Test Accuracy', test_acc)
    writer.add_scalar('Test Balance Accuracy', test_bacc)
    writer.add_scalar('Test F1-score', test_f1)
    writer.add_scalar('Test Weighted F1-score', test_weighted_f1)
    writer.add_scalar('Test Precision', test_precision)
    writer.add_scalar('Test Recall', test_recall)
    writer.add_scalar('Test AUROC', test_auroc)
    writer.add_scalar('Test AUPRC', test_auprc)
    writer.add_scalar('Test Kappa', test_kappa)
    
    return test_acc,test_bacc,test_f1,test_weighted_f1,test_precision,test_recall,test_auroc,test_auprc,test_kappa

def main():
    args = argparser.parse_args()
    print(args)

    # NCT-CRC-HE-100K (nonorm)
    root_pth = './NCT-CRC-HE-100K'
    
    # Load encoded lables
    with open(os.path.join(root_pth, 'split_index', 'crc100knonorm_train_labels_encoded.pkl'), 'rb') as f:
        crc100knonorm_train_labels_encoded = pickle.load(f)
    with open(os.path.join(root_pth, 'split_index', 'crc100knonorm_test_labels_encoded.pkl'), 'rb') as f:
        crc100knonorm_test_labels_encoded = pickle.load(f)

    train_features_list = []
    test_features_list = []
    for feature_type in args.features_list:
        with open(os.path.join(root_pth, feature_type, f'crc100knonorm_train_{feature_type}.pkl'), 'rb') as f:
            crc100knonorm_train = pickle.load(f)
        with open(os.path.join(root_pth, feature_type, f'crc100knonorm_val_{feature_type}.pkl'), 'rb') as f:
            crc100knonorm_test = pickle.load(f)
        train_features_list.append(crc100knonorm_train['embeddings'])
        test_features_list.append(crc100knonorm_test['embeddings'])
        
    concat_train_features = np.hstack(train_features_list)
    concat_test_features = np.hstack(test_features_list)

    # Construct training/test dataset
    train_feats = torch.Tensor(concat_train_features)
    train_labels = torch.Tensor(crc100knonorm_train_labels_encoded).type(torch.long)

    test_feats = torch.Tensor(concat_test_features)
    test_labels = torch.Tensor(crc100knonorm_test_labels_encoded).type(torch.long)
    
    train_dataset = CustomDataset(train_feats, train_labels)
    test_dataset = CustomDataset(test_feats, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    test_results_df = pd.DataFrame(columns=[
        'Accuracy', 'Balance_Accuracy', 'F1-score', 'Weighted_F1-score', 'Precision', 'Recall', 'AUROC', 'AUPRC', 'Kappa'
    ])
    
    for rep_record in range(args.rep_num):
        # print("rep_record:", rep_record)
        output_dir = os.path.join(args.output_root,'record' ,'rep_'+str(rep_record))
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
    
        train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset, replacement=True)
        train_sampler_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=train_sampler, pin_memory=True)
        infinite_train_loader = itertools.cycle(train_sampler_loader)

        model = LinearProbe(args.embed_dim, args.num_heads, args.encoding_size, 9)
        test_results = train_eval(model, infinite_train_loader, train_loader, test_loader, output_dir, args)
        test_results_df.loc[rep_record] = test_results
        
    mean_row = test_results_df.mean().to_frame().T
    std_row = test_results_df.std().to_frame().T
    mean_row.index = ['mean']
    std_row.index = ['std']

    test_results_df_all = pd.concat([test_results_df, mean_row, std_row])
    test_results_df_all.to_csv(os.path.join(args.output_root,'test_all_results.csv'), float_format='%.4f')
    
# Make argparser
argparser = argparse.ArgumentParser(description='Linear Probe')
argparser.add_argument('--features_list',       type=str, nargs='+', help='List of multiple FM features')
argparser.add_argument('--rep_num',             type=int, default=10, help='The number of the repetition')
argparser.add_argument('--embed_dim',           type=int, default=1024, help='The dimension of the embeddings')
argparser.add_argument('--num_heads',           type=int, default=1, help='The number of attention head')
argparser.add_argument('--encoding_size',       type=int, default=1280, help='The number of encoding size in cross attention mechanism')
argparser.add_argument('--batch_size',          type=int, default=512, help='Batch size')
argparser.add_argument('--train_iters',         type=int, default=10000, help='Number of iterations')
argparser.add_argument('--n_cycles',            type=int, default=1, help='Number of CosineAnnealingLR cycles')
argparser.add_argument('--lr',                  type=float, default=1e-4, help='Learning rate')
argparser.add_argument('--min_lr',              type=float, default=0.0, help='Minimum learning rate')
argparser.add_argument('--weight_decay',        type=float, default=0.0, help='Weight decay')
argparser.add_argument('--eval_interval',       type=int, default=500, help='Evaluation interval')
argparser.add_argument('--num_workers',         type=int, default=10, help='Number of workers')
argparser.add_argument('--output_root',         type=str, default='outputs', help='Output root path')

if __name__ == '__main__':
    main()