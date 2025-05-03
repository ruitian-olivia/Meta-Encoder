import os
import h5py
import multiprocessing
import numpy as np
import pandas as pd

from functools import reduce

def feature_integration(slide_id):
    data_dir_root = "./BRCA"
    features_list = ["FEATURES_DIRECTORY_CHIEF", "FEATURES_DIRECTORY_gigapath", "FEATURES_DIRECTORY_UNI"]

# for slide_id in slide_data['slide_id']:
    features_df_list = []
    for features_type in features_list:
        features_data_dir = os.path.join(data_dir_root, features_type)
        full_path = os.path.join(features_data_dir,'h5_files','{}.h5'.format(slide_id.split('.')[0]))
        with h5py.File(full_path,'r') as hdf5_file:
            features = hdf5_file['features'][:]
            coords = hdf5_file['coords'][:]
        feature_columns = [f'{features_type}_{i+1}' for i in range(features.shape[1])]
        coords_columns = [f'coords{i+1}' for i in range(coords.shape[1])]
        
        features_df = pd.DataFrame(np.hstack([features, coords]), columns=feature_columns + coords_columns)
        features_df_list.append(features_df)
        
    merged_features_df = reduce(lambda left, right: pd.merge(left, right, on=['coords1', 'coords2'], how='inner'), features_df_list)
    
    if merged_features_df.empty:
        print(f"DataFrame for {slide_id} is empty")
    else:
        saved_dir = os.path.join(data_dir_root, 'FEATURES_CHIEF_gigapath_UNI')
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)
        saved_file = os.path.join(saved_dir, "{}.h5".format(slide_id.split('.')[0]))
        with h5py.File(saved_file, 'w') as f:
            
            coords_df = merged_features_df[['coords1', 'coords2']]
            coords = coords_df.to_numpy(dtype=np.int64)
            
            f.create_dataset('coords', data=coords)
            
            for features_type in features_list:
                cols = [col for col in merged_features_df.columns if col.startswith(features_type)]

                selected_features_df = merged_features_df[cols]

                selected_features = selected_features_df.to_numpy(dtype=np.float32)
                f.create_dataset(features_type, data=selected_features)

if __name__ == "__main__":
    csv_path = "./subtype_list/tcga_brca_subset.csv"
    slide_data = pd.read_csv(csv_path)  

    with multiprocessing.Pool(processes=4) as pool:
        results = pool.map(feature_integration, slide_data['slide_id'])
        
   