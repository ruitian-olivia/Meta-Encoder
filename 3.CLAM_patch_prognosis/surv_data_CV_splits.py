import os
import json
import pathlib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

split_root = './splits_three_fold'
surv_df_filter = pd.read_csv('BRCA_surv_label.csv')

patient_surv_df = surv_df_filter[['TCGA_case_id', 'Days', 'Censor']]
patient_surv_df = patient_surv_df.drop_duplicates()

TCGA_case_id = patient_surv_df['TCGA_case_id']
vital_status = patient_surv_df['Censor']

for version_index in range(10):
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=version_index)
    splits_folds = {}
    for fold_idx, (train_index, test_index) in enumerate(skf.split(TCGA_case_id, vital_status)):
        train_case_ids = TCGA_case_id.iloc[train_index]
        test_case_ids = TCGA_case_id.iloc[test_index]
        
        splits_folds[f'fold_{fold_idx + 1}'] = {
            'train_case_ids': train_case_ids.tolist(),
            'test_case_ids': test_case_ids.tolist()
        }
            
    with open(f'./splits_json_file/BRCA_surv_FF_splits_{version_index}.json', 'w') as f:
        json.dump(splits_folds, f, indent=4)
 
fold_list = ['fold_1', 'fold_2', 'fold_3']
for version_index in range(10):

    with open(f'./splits_json_file/BRCA_surv_FF_splits_{version_index}.json', 'r') as f:
        folds_dict = json.load(f)
        
    for fold_name in fold_list:
        print(f'Vesion: {version_index}; Fold: {fold_name}')

        train_ids = folds_dict[fold_name]['train_case_ids']
        test_ids = folds_dict[fold_name]['test_case_ids']
        
        surv_df_train = surv_df_filter[surv_df_filter['TCGA_case_id'].isin(train_ids)]
        surv_df_test = surv_df_filter[surv_df_filter['TCGA_case_id'].isin(test_ids)]
        
        train_HE_id = surv_df_train['HE_entity_submitter_id'].to_list()
        test_HE_id = surv_df_test['HE_entity_submitter_id'].to_list()
        
        rows = list(zip_longest(train_HE_id, test_HE_id, fillvalue=''))
        df = pd.DataFrame(rows, columns=['train', 'test'])

        df.to_csv(f'splits_csv_file/Version_{version_index}_{fold_name}_split.csv')
