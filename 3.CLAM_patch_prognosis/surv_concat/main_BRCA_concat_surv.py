from __future__ import print_function

import os
import pdb
import math
import torch
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

from utils.file_utils import save_pkl, load_pkl
from utils.utils_survival import *
from utils.core_utils_survival import train
from datasets_CLAM.dataset_generic import Generic_MIL_Dataset
from datasets_CLAM.dataset_multiple_generic import Generic_MIL_Multiple_Dataset
from torch.utils.data import DataLoader, sampler
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    all_test_Cindex = []
    all_test_AUC = []
    version_list = []
    fold_item_list = []
    p_value_list = []

    fold_list = ['fold_1', 'fold_2', 'fold_3']
    
    for version_index in range(10):
        print("Version index:", version_index)
        for fold_index in fold_list:
            print("Fold index:", fold_index)
            seed_torch(args.seed)
            train_dataset, test_dataset = dataset.return_splits(
                    csv_path='{}/Version_{}_{}_split.csv'.format(args.split_dir, version_index, fold_index))
            
            print("train_dataset:", train_dataset)
            print("test_dataset:", test_dataset)
            
            datasets = (train_dataset, test_dataset)
            
            test_c_index, test_auc, test_KM_df = train(datasets, version_index, fold_index, args)
            
            predicted_risk_path = os.path.join(args.results_dir, 'predicted_risk')
            if not os.path.exists(predicted_risk_path):
                os.makedirs(predicted_risk_path)
            
            test_KM_df.to_csv(os.path.join(predicted_risk_path, f'Version_{version_index}_{fold_index}_risk_data.csv'))
            
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
            plt.savefig(os.path.join(predicted_risk_path, f'Version_{version_index}_{fold_index}_KM_plot.png'))

            all_test_Cindex.append(test_c_index)
            all_test_AUC.append(test_auc)
            version_list.append(version_index)
            fold_item_list.append(fold_index)
            p_value_list.append(results.p_value)
            

    final_df = pd.DataFrame({'Version': version_list, 'fold': fold_item_list,
        'test_Cindex': all_test_Cindex, 'test_3year_AUC': all_test_AUC, 'p_value': p_value_list})
    
    C_index_mean = final_df['test_Cindex'].mean()
    year_AUC3_mean = final_df['test_3year_AUC'].mean()
    
    C_index_std = final_df['test_Cindex'].std()
    year_AUC3_std = final_df['test_3year_AUC'].std()
    
    sig_perc1 = (final_df['p_value'] <= 0.05).mean()
    sig_perc2 = (final_df['p_value'] <= 0.1).mean()
    
    average_row = pd.DataFrame([['average', 'average', C_index_mean, year_AUC3_mean, sig_perc1], \
        ['std', 'std', C_index_std, year_AUC3_std, sig_perc2]], columns=['Version', 'fold', 'test_Cindex', 'test_3year_AUC', 'p_value'])
    summary_test_df = pd.concat([final_df, average_row], ignore_index=True)

    summary_test_df.to_csv(os.path.join(args.results_dir, 'summary_results.csv'), index=False, float_format='%.4f')
        
# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--max_epochs', type=int, default=50,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', action='store_true', default=False, help='enable dropout (p=0.25)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='svm',
                     help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb', 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping', 'task_3_tumor_survival'])
### CLAM specific options
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False, 
                     help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')
### Training dataset selection
parser.add_argument('--encoding_size', type=int, default=1024, help='Encoding size of patches')
parser.add_argument('--csv_path', type=str, default=None, 
                    help='csv file path for training dataset')
### Multiple FM features
parser.add_argument('--features_list', type=str, nargs='+', help='List of multiple FM features')

args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = args.encoding_size
settings = {'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}

if args.model_type in ['clam_sb', 'clam_mb']:
   settings.update({'bag_weight': args.bag_weight,
                    'inst_loss': args.inst_loss,
                    'B': args.B})

print('\nLoad Dataset')

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = args.csv_path,
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal':0, 'tumor':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = args.csv_path,
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'IDC':0, 'ILC':1},
                            patient_strat= False,
                            ignore=[])

    if args.model_type in ['clam_sb', 'clam_mb']:
        assert args.subtyping

elif args.task == 'task_3_tumor_survival':
    args.n_classes=2
    dataset = Generic_MIL_Multiple_Dataset(csv_path = args.csv_path,
                            data_dir= args.data_root_dir,
                            features_list = args.features_list,
                            shuffle = False, 
                            seed = args.seed, 
                            ignore=[]) 
        
else:
    raise NotImplementedError
    
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.split_dir = os.path.join('splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")


