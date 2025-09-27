from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle
from scipy import stats

from torch.utils.data import Dataset
import h5py

from utils.utils import generate_split, nth

from functools import reduce

def save_splits(split_datasets, column_keys, filename, boolean_style=False):
    splits = [split_datasets[i].slide_data['HE_entity_submitter_id'] for i in range(len(split_datasets))]
    if not boolean_style:
        df = pd.concat(splits, ignore_index=True, axis=1)
        df.columns = column_keys
    else:
        df = pd.concat(splits, ignore_index = True, axis=0)
        index = df.values.tolist()
        one_hot = np.eye(len(split_datasets)).astype(bool)
        bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
        df = pd.DataFrame(bool_array, index=index, columns = ['train', 'test'])

    df.to_csv(filename)
    print()

class Generic_WSI_Classification_Dataset(Dataset):
    def __init__(self,
        csv_path = 'dataset_csv/ccrcc_clean.csv',
        shuffle = False, 
        seed = 7, 
        ignore=[],
        patient_strat=False,

        patient_voting = 'max',
        ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            shuffle (boolean): Whether to shuffle
            seed (int): random seed for shuffling the data
            print_info (boolean): Whether to print a summary of the dataset
            label_dict (dict): Dictionary with key, value pairs for converting str labels to int
            ignore (list): List containing class labels to ignore
        """

        self.seed = seed
        self.train_ids, self.test_ids = (None, None)
        self.data_dir = None

        slide_data = pd.read_csv(csv_path)
        print("slide_data.head():", slide_data.head())

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)

        self.slide_data = slide_data

        self.patient_data_prep()

    def patient_data_prep(self):
        patients = np.unique(np.array(self.slide_data['TCGA_case_id'])) # get unique patients
        patient_days = []
        patient_censor = []
        
        for p in patients:
            locations = self.slide_data[self.slide_data['TCGA_case_id'] == p].index.tolist()
            assert len(locations) > 0
            days = self.slide_data['Days'][locations].values
            censor = self.slide_data['Censor'][locations].values

            days = stats.mode(days)[0]
            censor = stats.mode(censor)[0]

            patient_days.append(days)
            patient_censor.append(censor)
        
        self.patient_data = {'TCGA_case_id':patients, 'Days':np.array(patient_days), 'Censor': np.array(patient_censor)}


    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['TCGA_case_id'])

        else:
            return len(self.slide_data)

    def create_splits(self, k = 3, val_num = (25, 25), test_num = (40, 40), label_frac = 1.0, custom_test_ids = None):
        settings = {
                    'n_splits' : k, 
                    'val_num' : val_num, 
                    'test_num' : test_num,
                    'label_frac' : label_frac,
                    'seed' : self.seed,
                    'custom_test_ids' : custom_test_ids
                    }

        if self.patient_strat:
            settings.update({'cls_ids' : self.patient_cls_ids, 'samples': len(self.patient_data['case_id'])})
        else:
            settings.update({'cls_ids' : self.slide_cls_ids, 'samples': len(self.slide_data)})

        self.split_gen = generate_split(**settings)
        
    def set_splits(self,start_from=None):
        if start_from:
            ids = nth(self.split_gen, start_from)

        else:
            ids = next(self.split_gen)

        if self.patient_strat:
            slide_ids = [[] for i in range(len(ids))] 

            for split in range(len(ids)): 
                for idx in ids[split]:
                    case_id = self.patient_data['TCGA_case_id'][idx]
                    slide_indices = self.slide_data[self.slide_data['TCGA_case_id'] == case_id].index.tolist()
                    slide_ids[split].extend(slide_indices)

            self.train_ids, self.test_ids = slide_ids[0], slide_ids[1]

        else:
            self.train_ids, self.test_ids = ids
        
    def get_split_from_df(self, all_splits, split_key='train'):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)
        split_list_ID = split.tolist()
        mask = self.slide_data['HE_entity_submitter_id'].isin(split_list_ID)
        df_slice = self.slide_data[mask].reset_index(drop=True)
        split = Generic_Split(df_slice, self.features_list, data_dir=self.data_dir)
        
        return split

    def return_splits(self, csv_path):

        all_splits = pd.read_csv(csv_path, dtype=self.slide_data['HE_entity_submitter_id'].dtype)  # Without "dtype=self.slide_data['slide_id'].dtype", read_csv() will convert all-number columns to a numerical type. Even if we convert numerical columns back to objects later, we may lose zero-padding in the process; the columns must be correctly read in from the get-go. When we compare the individual train/val/test columns to self.slide_data['slide_id'] in the get_split_from_df() method, we cannot compare objects (strings) to numbers or even to incorrectly zero-padded objects/strings. An example of this breaking is shown in https://github.com/andrew-weisman/clam_analysis/tree/main/datatype_comparison_bug-2021-12-01.
        train_split = self.get_split_from_df(all_splits, 'train')
        test_split = self.get_split_from_df(all_splits, 'test')
            
        return train_split, test_split

    def get_list(self, ids):
        return self.slide_data['HE_entity_submitter_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['Days'][ids], self.slide_data['Censor'][ids]

    def __getitem__(self, idx):
        return None

        
class Generic_MIL_Multiple_Dataset(Generic_WSI_Classification_Dataset):
    def __init__(self,
        data_dir, 
        features_list,
        **kwargs):
    
        super(Generic_MIL_Multiple_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.data_dir_root = data_dir
        self.features_list = features_list

    def __getitem__(self, idx):
        slide_id = self.slide_data['HE_entity_submitter_id'][idx]
        days = self.slide_data['Days'][idx]
        censor = self.slide_data['Censor'][idx]
        data_dir_root = self.data_dir_root
        features_list = self.features_list

        full_path = os.path.join(data_dir_root, '{}.h5'.format(slide_id.split('.')[0]))
        with h5py.File(full_path,'r') as hdf5_file:
            coords = hdf5_file['coords'][:]
            
            features_np_list = []
            for features_type in features_list:
                features = hdf5_file[features_type][:]
                features_np_list.append(features)

        features = torch.from_numpy(np.hstack(features_np_list))
        
        return features, days, censor, coords


class Generic_Split(Generic_MIL_Multiple_Dataset):
    def __init__(self, slide_data, features_list, data_dir=None, num_classes=2):
        self.use_h5 = False
        self.slide_data = slide_data
        self.features_list = features_list
        self.data_dir_root = data_dir
        self.data_dir = os.path.join(data_dir, self.features_list[0])
        self.num_classes = num_classes

    def __len__(self):
        return len(self.slide_data)
        


