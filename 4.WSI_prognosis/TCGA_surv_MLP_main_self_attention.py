import os
import json
import h5py
import glob
import torch
import argparse
import numpy as np
import pandas as pd
import torch.utils.tensorboard as tensorboard
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from lifelines.utils import concordance_index
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from attention_utils import MultiheadAttention


def getWSIFeatures(HE_id, WSI_root):
    WSI_path = os.path.join(WSI_root, f'{HE_id}.pt')
    if os.path.isfile(WSI_path):
        return WSI_path
    else:
        return np.nan     

def getDays(vital_status, days_to_death, days_to_last_follow_up):
    if vital_status == "Alive":
        days = days_to_last_follow_up
    elif vital_status == "Dead":
        days = days_to_death
    else:
        return np.nan
    
    if '--' not in days:
        return int(float(days))
    else:
        return np.nan
    
def getCensor(vital_status):
    if vital_status == "Alive":
        return 0
    elif vital_status == "Dead":
        return 1
    else:
        return np.nan

class cox_ph_loss(torch.nn.Module):

    def __init__(self):
        super(cox_ph_loss, self).__init__()

    def forward(self, y_true_time, y_true_event, y_pred_hr):
        _, idx = torch.sort(-y_true_time)
        y_true_time = y_true_time[idx]
        y_true_event = y_true_event[idx]
        y_pred_hr = y_pred_hr[idx]
        
        risk_score = torch.exp(y_pred_hr)
        log_risk = torch.log(torch.cumsum(risk_score, dim=0))
        uncensored_likelihood = y_pred_hr - log_risk
        censored_likelihood = uncensored_likelihood * y_true_event
        
        cox_loss = -torch.sum(censored_likelihood) / y_true_time.shape[0]

        return cox_loss

class SurvDataset(Dataset):
    def __init__(self, surv_df):
        super(SurvDataset, self).__init__()
        self.surv_df = surv_df

    def __len__(self):
        return len(self.surv_df)

    def __getitem__(self, idx):
        item_name = self.surv_df['HE_entity_submitter_id'].tolist()[idx]
        match_item = self.surv_df[self.surv_df["HE_entity_submitter_id"] == item_name]
        
        WSI_features_item1 = torch.load(match_item['WSI_features_path1'].tolist()[0]) # filelist记录pt文件的位置
        WSI_features_item2 = torch.load(match_item['WSI_features_path2'].tolist()[0]) # filelist记录pt文件的位置

        survival_item = match_item['Days'].tolist()[0]
        censor_item = match_item['Censor'].tolist()[0]
        
        return WSI_features_item1, WSI_features_item2, survival_item, censor_item

class MLP_ProbeSurv(torch.nn.Module):

    def __init__(self, embed_dim, num_heads, feature_num, latent_dim):
        super(MLP_ProbeSurv, self).__init__()
        
        self.cross_attention = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, kdim=feature_num, vdim=feature_num, qdim=feature_num)

        self.linear = torch.nn.Linear(embed_dim, latent_dim)
        self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(latent_dim, 2)
        self.risk_score = torch.nn.Linear(2, 1)
        
    def forward(self, x1, x2):
        x =  torch.cat([x1, x2], dim=-1)
        x = x.squeeze()
        x = x.unsqueeze(0)
        x, _ = self.cross_attention(x, x, x)
        x = x.squeeze()
        
        x = self.relu(self.linear(x))
        x = self.fc(x)
        x = F.softmax(x, dim=-1)
        x = self.risk_score(x)
                
        return x

def evaluate(model, criterion, val_loader, val_case_id, device, args):
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        pred_risks, survival_gather, censor_gather = None, None, None
        for _, batch in enumerate(val_loader):
            WSI_features1, WSI_features2, survival, censor = batch
            WSI_features1, WSI_features2, survival, censor = WSI_features1.to(device), WSI_features2.to(device), survival.to(device), censor.to(device)
            
            output = model(WSI_features1, WSI_features2)
            
            if pred_risks is None:
                pred_risks = output.cpu().numpy()
                survival_gather = survival.cpu().numpy()
                censor_gather = censor.cpu().numpy()
            else:
                pred_risks = np.concatenate((pred_risks, output.cpu().numpy()), axis=0)
                survival_gather = np.concatenate((survival_gather, survival.cpu().numpy()), axis=0)
                censor_gather = np.concatenate((censor_gather, censor.cpu().numpy()), axis=0)
        
    pred_risks_df = pd.DataFrame(
        data = np.squeeze(pred_risks),
        index = val_case_id,
        columns = ['Predicted_risk']
    )

    target_surv_df = pd.DataFrame(
        data = np.column_stack((survival_gather, censor_gather)),
        index = val_case_id,
        columns = ['Survival', 'Censor']
    )
    
    pred_agg_risks_df = pred_risks_df.groupby(pred_risks_df.index).max()
    target_agg_surv_df = target_surv_df.groupby(target_surv_df.index).first()

    val_loss = criterion(torch.tensor(target_agg_surv_df['Survival']), torch.tensor(target_agg_surv_df['Censor']), torch.tensor(pred_agg_risks_df['Predicted_risk']))
    c_index = concordance_index(target_agg_surv_df['Survival'], -np.exp(pred_agg_risks_df['Predicted_risk']), target_agg_surv_df['Censor'])
    
    return val_loss, c_index, target_agg_surv_df

def test(model, test_loader, test_case_id, train_surv_df, device, args):
    # Evaluate the model
    model.eval()
    with torch.no_grad():

        pred_risks, survival_gather, censor_gather = None, None, None
        for _, batch in enumerate(test_loader):
            WSI_features1, WSI_features2, survival, censor = batch
            WSI_features1, WSI_features2, survival, censor = WSI_features1.to(device), WSI_features2.to(device), survival.to(device), censor.to(device)
            
            output = model(WSI_features1, WSI_features2)

            if pred_risks is None:
                pred_risks = output.cpu().numpy()
                survival_gather = survival.cpu().numpy()
                censor_gather = censor.cpu().numpy()
            else:
                pred_risks = np.concatenate((pred_risks, output.cpu().numpy()), axis=0)
                survival_gather = np.concatenate((survival_gather, survival.cpu().numpy()), axis=0)
                censor_gather = np.concatenate((censor_gather, censor.cpu().numpy()), axis=0)
    
    pred_risks_df = pd.DataFrame(
        data = np.squeeze(pred_risks),
        index = test_case_id,
        columns = ['Predicted_risk']
    )

    target_surv_df = pd.DataFrame(
        data = np.column_stack((survival_gather, censor_gather)),
        index = test_case_id,
        columns = ['Survival', 'Censor']
    )

    pred_agg_risks_df = pred_risks_df.groupby(pred_risks_df.index).max()
    target_agg_surv_df = target_surv_df.groupby(target_surv_df.index).first()
    
    test_c_index = concordance_index(target_agg_surv_df['Survival'], -np.exp(pred_agg_risks_df['Predicted_risk']), target_agg_surv_df['Censor'])
    
    train_X_structured = Surv.from_dataframe('Censor', 'Survival', train_surv_df)
    test_X_structured = Surv.from_dataframe('Censor', 'Survival', target_agg_surv_df)
    
    times = [365, 730, 1095]
    test_aucs, test_mean_auc = cumulative_dynamic_auc(train_X_structured, test_X_structured, np.exp(pred_agg_risks_df['Predicted_risk']), times)
    
    return test_c_index, test_aucs, test_mean_auc, pred_agg_risks_df, target_agg_surv_df
    
def train_eval(model,
          train_loader,
          val_loader,
          test_loader,
          val_case_id,
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
    
    criterion = cox_ph_loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Set the learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_num*len(train_loader) // args.n_cycles , eta_min=args.min_lr)
    
    for epoch in range(args.epochs_num):
        for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs_num}"):
            WSI_features1, WSI_features2, survival, censor = batch
            WSI_features1, WSI_features2,  survival, censor = WSI_features1.to(device), WSI_features2.to(device), survival.to(device), censor.to(device)
            
            output = model(WSI_features1, WSI_features2)
            
            loss = criterion(survival, censor, output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', lr, epoch)
        train_loss, train_c_index, train_surv_df = evaluate(model, criterion, val_loader, val_case_id, device, args)
        writer.add_scalar('Train Loss', train_loss, epoch)
        writer.add_scalar('Train C-index', train_c_index, epoch)      

    test_c_index, test_aucs, test_mean_auc, pred_agg_risks_df, target_agg_surv_df = test(model, test_loader, test_case_id, train_surv_df, device, args)
    
    writer.add_scalar(f'Test C-index', test_c_index)
    writer.add_scalar(f'Test 3-year AUC', test_aucs[2])
    
    test_KM_df = pd.merge(pred_agg_risks_df, target_agg_surv_df, left_index=True, right_index=True, how='inner')
    
    return test_c_index, test_aucs, test_mean_auc, test_KM_df

def main():
    args = argparser.parse_args()
    print(args)
    
    WSI_features_root = '../2.WSI_subtyping/embedding_features/BRCA'
    surv_csv_path = 'BRCA_surv.csv'
    split_root = '../3.CLAM_patch_prognosis/splits_json_file'
    
    WSI_features_path = os.path.join(WSI_features_root, args.cancer_type)
    WSI_features_type_path1 = os.path.join(WSI_features_path, f'FEATURES_DIRECTORY_{args.feature_list[0]}_WSI_features')
    WSI_features_type_path2 = os.path.join(WSI_features_path, f'FEATURES_DIRECTORY_{args.feature_list[1]}_WSI_features')
    
    output_path = os.path.join(args.output_root, args.cancer_type, args.version)
        
    feature_num = 2048
    surv_df = pd.read_csv(surv_csv_path)
    surv_df.rename(columns={'demographic.vital_status': 'vital_status', \
                            'demographic.days_to_death': 'days_to_death', \
                            'diagnoses.days_to_last_follow_up': 'days_to_last_follow_up', \
                            'project.project_id': 'project_id'}, inplace=True)

    surv_df['WSI_features_path1'] = surv_df.apply(lambda x: getWSIFeatures(x.HE_entity_submitter_id, WSI_features_type_path1), axis = 1)
    surv_df['WSI_features_path2'] = surv_df.apply(lambda x: getWSIFeatures(x.HE_entity_submitter_id, WSI_features_type_path2), axis = 1)   
    surv_df['Days'] = surv_df.apply(lambda x: getDays(x.vital_status, x.days_to_death, x.days_to_last_follow_up), axis = 1)
    surv_df['Censor'] = surv_df.apply(lambda x: getCensor(x.vital_status), axis = 1)
    
    surv_select_df = surv_df[['TCGA_case_id', 'HE_entity_submitter_id', 'WSI_features_path1', 'WSI_features_path2','Days', 'Censor']]
    print('surv_select_df.shape:', surv_select_df.shape)
    surv_df_filter = surv_select_df.dropna()
    
    surv_df_filter = surv_df_filter[surv_df_filter['Days'] >= 0]
    print('surv_df_filter.shape:', surv_df_filter.shape)
    surv_df_censor = surv_df_filter.loc[surv_df_filter['Censor'] == 1]
    print("All censored number:", len(surv_df_censor))
    surv_df_event = surv_df_filter.loc[surv_df_filter['Censor'] == 0]
    print("All death number", len(surv_df_event))
        
    fold_list = ['fold_1', 'fold_2', 'fold_3']
    
    test_results_list = []
    
    for version_index in range(10):
    
        with open(os.path.join(split_root, f'{args.cancer_type}_surv_FF_splits_{version_index}.json'), 'r') as f:
            folds_dict = json.load(f)
            
        for fold_name in fold_list:
            print(f'Vesion: {version_index}; Fold: {fold_name}')
            
            train_ids = folds_dict[fold_name]['train_case_ids']
            test_ids = folds_dict[fold_name]['test_case_ids']
            
            surv_df_train = surv_df_filter[surv_df_filter['TCGA_case_id'].isin(train_ids)]
            surv_df_test = surv_df_filter[surv_df_filter['TCGA_case_id'].isin(test_ids)]
            
            val_case_id = surv_df_train['TCGA_case_id'].values
            test_case_id = surv_df_test['TCGA_case_id'].values
            
            train_dataset = SurvDataset(surv_df=surv_df_train)
            test_dataset = SurvDataset(surv_df=surv_df_test)
            print("len(train_dataset):", len(train_dataset))
            print("len(test_dataset):", len(test_dataset))
            
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(train_dataset, batch_size=args.batch_size)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
                        
            model = MLP_ProbeSurv(args.embed_dim,  args.num_heads, feature_num, args.latent_dim,)
            
            output_dir = os.path.join(args.output_root, args.cancer_type, args.version, f"{version_index}_{fold_name}")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            test_c_index, test_aucs, _, test_KM_df = train_eval(model, train_loader, val_loader, test_loader, val_case_id, test_case_id, output_dir, args)
            print("test C-index:", test_c_index)
            print("test 3-year AUC:", test_aucs[2])
            
            test_KM_df.to_csv(os.path.join(output_path,f'Version_{version_index}_{fold_name}_risk_data.csv'))
        
            test_risk_scores = test_KM_df['Predicted_risk'].values
            test_censor_status = test_KM_df['Censor'].values
            test_days_surv = test_KM_df['Survival'].values
            median_risk = np.median(test_risk_scores)
            high_risk = test_risk_scores > median_risk
            low_risk = test_risk_scores <= median_risk
            
            kmf = KaplanMeierFitter()
            plt.figure(figsize=(8, 6))

            kmf.fit(test_days_surv[high_risk], event_observed=test_censor_status[high_risk], label=f"High Risk (n={sum(high_risk)})")
            ax = kmf.plot_survival_function(ci_show=True, linewidth=2, color='red')

            kmf.fit(test_days_surv[low_risk], event_observed=test_censor_status[low_risk], label=f"Low Risk (n={sum(low_risk)})")
            kmf.plot_survival_function(ax=ax, ci_show=True, linewidth=2, color='blue')

            results = logrank_test(
                test_days_surv[high_risk],
                test_days_surv[low_risk],
                test_censor_status[high_risk],
                test_censor_status[low_risk]
            )

            p_value_text = f"Log-rank p = {results.p_value:.3f}" if results.p_value >= 0.001 else "Log-rank p < 0.001"
            plt.text(0.5, 0.2, p_value_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

            plt.title("Kaplan-Meier Survival Analysis", fontsize=14)
            plt.xlabel("Survival Time (Days)", fontsize=12)
            plt.ylabel("Survival Probability", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend(loc='upper right', fontsize=10)

            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f'Version_{version_index}_{fold_name}_KM_plot.png'))
            
            test_result_record = {'version_index': version_index, 'fold_name':fold_name, 'C_index':test_c_index, '3-year AUC':test_aucs[2], 'p_value': results.p_value}
            test_results_list.append(test_result_record)

    test_results_df = pd.DataFrame(test_results_list)
    
    C_index_mean = test_results_df['C_index'].mean()
    year_AUC3_mean = test_results_df['3-year AUC'].mean()
    
    C_index_std = test_results_df['C_index'].std()
    year_AUC3_std = test_results_df['3-year AUC'].std()
    
    sig_perc1 = (test_results_df['p_value'] <= 0.05).mean()
    sig_perc2 = (test_results_df['p_value'] <= 0.1).mean()
    
    average_row = pd.DataFrame([['average', C_index_mean, year_AUC3_mean, sig_perc1], \
        ['std', C_index_std, year_AUC3_std, sig_perc2]], columns=['fold_name', 'C_index', '3-year AUC', 'p_value'])
    summary_test_df = pd.concat([test_results_df, average_row], ignore_index=True)
    
    summary_test_df.to_csv(os.path.join(output_path, 'test_results_summary.csv'), index=False, float_format='%.4f')
    
# Make argparser
argparser = argparse.ArgumentParser(description='Linear Probe Survival Analysis')
# argparser.add_argument('--feature_type',        type=str, default='TITAN', choices=['TITAN', 'PRISM'], help='Feature type')
argparser.add_argument('--feature_list',       type=str, nargs='+')
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
argparser.add_argument('--embed_dim',           type=int, default=1024, help='dimension of embedded features')
argparser.add_argument('--num_heads',           type=int, default=1, help='number of attention head')


if __name__ == '__main__':
    main()