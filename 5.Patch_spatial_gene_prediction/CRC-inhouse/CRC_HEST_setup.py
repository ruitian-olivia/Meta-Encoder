import os
import numpy as np
import pandas as pd
import scanpy as sc

from hest import VisiumReader

CRC_Visium_root = './dataset/Visium_format'
CRC_saved_root = './dataset/HEST_format'
CRC_sample_list = ['CRC_inhouse_1', 'CRC_inhouse_2', 'CRC_inhouse_3', 'CRC_inhouse_4', 'CRC_inhouse_5', 'CRC_inhouse_6', 'CRC_inhouse_7', 'CRC_inhouse_8', 'CRC_inhouse_9', 'CRC_inhouse_10']

for index, CRC_sample_id in enumerate(CRC_sample_list):
    print("CRC_sample_id:", CRC_sample_id)
    sample_name = f'CRC_inhouse_{index+1}'
    print("sample_name:", sample_name)
    fullres_img_path = os.path.join(CRC_Visium_root, CRC_sample_id, 'HE_image', 'histology.tiff') # tiff
    bc_matrix_path = os.path.join(CRC_Visium_root, CRC_sample_id, 'filtered_feature_bc_matrix.h5')
    spatial_coord_path= os.path.join(CRC_Visium_root, CRC_sample_id, 'spatial')
    
    st = VisiumReader().read(
        fullres_img_path, # path to a full res image
        bc_matrix_path, # path to filtered_feature_bc_matrix.h5
        spatial_coord_path=spatial_coord_path # path to a space ranger spatial/ folder containing either a tissue_positions.csv or tissue_position_list.csv
    )
    
    saved_path = os.path.join(CRC_saved_root, sample_name)
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
        
    st.save(saved_path, save_img=False)
    st.segment_tissue(method = 'deep')
    st.dump_patches(saved_path)