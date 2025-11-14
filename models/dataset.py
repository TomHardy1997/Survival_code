import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import logging
import argparse
import ast
import h5py
import numpy as np


class PrognosisDataset(Dataset):
    def __init__(self, df, h5_dir=None):
        self.df = pd.read_csv(df)
        self.patient = self.df["case_id"]
        
        # ✅ 将 gender 转换为数字编码
        gender_map = {'Male': 0, 'Female': 1, 'MALE': 0, 'FEMALE': 1, 
                      'male': 0, 'female': 1, 'M': 0, 'F': 1}
        self.gender = self.df["gender"].map(gender_map).fillna(0).astype(int)
        
        self.age = pd.to_numeric(self.df["age"], errors='coerce').fillna(-1)
        self.label = self.df["disc_label"]
        self.time = self.df['survival_months']
        self.censor = self.df['censorship']
        self.h5_dir = h5_dir
        self.wsi = self.df['slide_id'].apply(lambda x: ast.literal_eval(x.strip()))
        
        if h5_dir and not os.path.exists(h5_dir):
            raise ValueError(f"H5 directory not found: {h5_dir}")

    def __len__(self):
        return len(self.patient)

    def __getitem__(self, idx):
        patient = self.patient[idx]
        gender = int(self.gender[idx])  # ✅ 确保是整数
        age = float(self.age[idx])
        label = self.label[idx]
        sur_time = self.time[idx]
        censor = self.censor[idx]
        slide_ids = self.wsi[idx]
        
        if not slide_ids:
            raise ValueError(f"No slide_ids for patient {patient}")
        
        path_features = []
        path_coords = []
        
        for slide_id in slide_ids:
            h5_id = slide_id.strip().replace('.pt', '.h5')
            h5_path = os.path.join(self.h5_dir, h5_id)
            
            if not os.path.exists(h5_path):
                print(f'Warning: File not found {h5_path}')
                continue
            
            try:
                with h5py.File(h5_path, 'r') as f:
                    features = f['features'][:]
                    coords = f['coords'][:]
                    
                    if features.shape[0] != coords.shape[0]:
                        print(f'Warning: Shape mismatch in {h5_path}')
                        continue
                    
                    path_features.append(torch.tensor(features, dtype=torch.float32))
                    path_coords.append(torch.tensor(coords, dtype=torch.float32))
            except Exception as e:
                print(f'Error loading {h5_path}: {e}')
                continue

        if not path_features:
            print(f"Warning: No valid features for patient {patient}")
            return None
        
        features = torch.cat(path_features, dim=0)
        coords = torch.cat(path_coords, dim=0)
        num_patches = features.shape[0]
        
        return patient, gender, age, label, sur_time, censor, features, coords, num_patches





# if __name__ == '__main__':
#     df = './tcga_survival_matched.csv'
#     h5_dir = '/home/stat-jijianxin/PFMs/features_conch_v1'
#     dataset = PrognosisDataset(df, h5_dir)
    
#     # 查看第一个样本
#     patient, gender, age, label, sur_time, censor, features, coords, num_patches = dataset[0]
    
#     print(f"Patient: {patient}")
#     print(f"Features shape: {features.shape}")  # 应该是 (1895, 512)
#     print(f"Coords shape: {coords.shape}")      # 应该是 (1895, 2)
#     print(f"Num patches: {num_patches}")        # 应该是 1895
#     import ipdb;ipdb.set_trace()
#     # 查看原始h5文件的属性
#     slide_ids = dataset.wsi[0]
#     h5_path = os.path.join(h5_dir, slide_ids[0].strip().replace('.pt', '.h5'))
#     with h5py.File(h5_path, 'r') as f:
#         print(f"\nH5 file attributes:")
#         print(f"  encoder: {f['features'].attrs['encoder']}")
#         print(f"  patch_size: {f['coords'].attrs['patch_size']}")
#         print(f"  magnification: {f['coords'].attrs['target_magnification']}")
if __name__ == '__main__':
    df = './tcga_survival_matched.csv'
    h5_dir = '/home/stat-jijianxin/PFMs/TRIDENT/tcga_filtered/20x_512px_0px_overlap/features_conch_v15'
    dataset = PrognosisDataset(df, h5_dir)
    
    print("=" * 60)
    print("检查多个H5文件拼接情况")
    print("=" * 60)
    
    # 找一个有多个slide的患者
    for idx in range(len(dataset)):
        slide_ids = dataset.wsi[idx]
        if len(slide_ids) > 1:
            print(f"\n找到患者 {dataset.patient[idx]}，有 {len(slide_ids)} 个slides:")
            
            # 逐个查看每个h5文件
            individual_shapes = []
            for i, slide_id in enumerate(slide_ids):
                h5_id = slide_id.strip().replace('.pt', '.h5')
                h5_path = os.path.join(h5_dir, h5_id)
                
                with h5py.File(h5_path, 'r') as f:
                    feat_shape = f['features'].shape
                    coord_shape = f['coords'].shape
                    individual_shapes.append(feat_shape[0])
                    print(f"  Slide {i+1}: {os.path.basename(h5_path)}")
                    print(f"    Features: {feat_shape}, Coords: {coord_shape}")
            
            # 通过__getitem__获取拼接后的结果
            patient, gender, age, label, sur_time, censor, features, coords, num_patches = dataset[idx]
            
            print(f"\n拼接后:")
            print(f"  Features shape: {features.shape}")
            print(f"  Coords shape: {coords.shape}")
            print(f"  Num patches: {num_patches}")
            print(f"\n验证: {sum(individual_shapes)} (各文件patch数之和) == {num_patches} (拼接后总数)")
            print(f"  结果: {'✓ 正确' if sum(individual_shapes) == num_patches else '✗ 错误'}")
            
            break  # 只检查第一个多slide的患者
    else:
        print("\n未找到有多个slides的患者，检查单个slide的情况:")
        patient, gender, age, label, sur_time, censor, features, coords, num_patches = dataset[0]
        print(f"Patient: {dataset.patient[0]}")
        print(f"Slides: {dataset.wsi[0]}")
        print(f"Features shape: {features.shape}")
        print(f"Coords shape: {coords.shape}")