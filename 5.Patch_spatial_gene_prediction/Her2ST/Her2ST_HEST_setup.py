import os
import numpy as np
import pandas as pd
import scanpy as sc

from hest import STReader

Her2ST_root = './dataset/legacy_ST_format'
Her2ST_saved_root = './dataset/HEST_format'

for first_level_folder in os.listdir(Her2ST_root):
    first_level_path = os.path.join(Her2ST_root, first_level_folder)
    if os.path.isdir(first_level_path):
        print('Patient name:', first_level_folder)
        for second_level_folder in os.listdir(first_level_path):
            second_level_path = os.path.join(first_level_path, second_level_folder)
            if os.path.isdir(second_level_path):
                print("Sample name:", second_level_folder)
                
                merged_path = os.path.join(second_level_path, 'merged.csv')
                img_path = os.path.join(second_level_path, 'histology.tiff')
                
                st = STReader().read(
                    merged_path=merged_path, 
                    img_path=img_path, 
                    spot_diameter=100.,
                    inter_spot_dist=200.
                )
                    
                saved_path = os.path.join(Her2ST_saved_root, first_level_folder, second_level_folder)
                if not os.path.exists(saved_path):
                    os.makedirs(saved_path)
                    
                st.save(saved_path, save_img=False)
                st.segment_tissue(method = 'deep')
                st.dump_patches(saved_path)