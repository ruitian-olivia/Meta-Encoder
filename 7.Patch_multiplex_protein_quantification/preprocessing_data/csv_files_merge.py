import os
import pandas as pd
import numpy as np

ORION_CRC_root_path = '../Orion-CRC'

slide_df = pd.read_csv(os.path.join(ORION_CRC_root_path, 'slide_dataframe.csv'))
slide_patient_df = slide_df[['in_slide_name', 'orion_slide_id']]

df_train = pd.read_csv(os.path.join(ORION_CRC_root_path, 'train_dataframe.csv'))
df_val = pd.read_csv(os.path.join(ORION_CRC_root_path, 'val_dataframe.csv'))
df_test = pd.read_csv(os.path.join(ORION_CRC_root_path, 'test_dataframe.csv'))

df_combined = pd.concat([df_test, df_train, df_val], axis=0)
df_combined['HE_path'] = df_combined['image_path'].str.replace('^he/', '', regex=True)

nmzd_HE_list = os.listdir(os.path.join(ORION_CRC_root_path, 'ORION_dataset_20x_he_norm'))
HE_df_combined = df_combined[df_combined['HE_path'].isin(nmzd_HE_list)]

HE_patient_df_combined = pd.merge(HE_df_combined, slide_patient_df, on='in_slide_name', how='left')
HE_patient_df_combined.to_csv(os.path.join(ORION_CRC_root_path, 'ORION_HE_dataframe.csv'), index=False)
