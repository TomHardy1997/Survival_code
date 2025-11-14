import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np
import h5py
import ast
from sklearn.model_selection import StratifiedKFold

class PrognosisDataset(Dataset):
    """
    生存预后数据集 - 支持患者级别分层、K-fold交叉验证、缓存等功能
    专门用于已提取特征的H5文件
    """
    def __init__(self, 
                 csv_path,
                 h5_dir,
                 label_col='disc_label',
                 shuffle=False,
                 seed=42,
                 use_cache=True,
                 print_info=True):
        """
        Args:
            csv_path: CSV文件路径
            h5_dir: H5特征文件目录
            label_col: 标签列名
            shuffle: 是否打乱数据
            seed: 随机种子
            use_cache: 是否使用缓存
            print_info: 是否打印数据集信息
        """
        self.seed = seed
        self.use_cache = use_cache
        self.print_info = print_info
        self.label_col = label_col
        
        # 数据缓存
        self.data_cache = {}
        
        # 加载CSV数据
        self.slide_data = pd.read_csv(csv_path)
        
        # 预处理数据
        self._preprocess_data()
        
        # 设置H5目录
        self.h5_dir = h5_dir
        if not os.path.exists(h5_dir):
            raise ValueError(f"H5 directory not found: {h5_dir}")
        
        # 构建患者-切片映射
        self._build_patient_mapping()
        
        # 准备患者级别数据
        self._prepare_patient_data()
        
        # 准备类别索引 (用于类别平衡采样)
        self._prepare_class_indices()
        
        # 初始化分割
        self.train_ids = []
        self.val_ids = []
        self.test_ids = []
        
        if shuffle:
            np.random.seed(seed)
            patient_indices = np.arange(self.num_patients)
            np.random.shuffle(patient_indices)
            # 重新排列patient_data
            for key in self.patient_data:
                self.patient_data[key] = self.patient_data[key][patient_indices]
        
        if print_info:
            self.summarize()
    
    def _preprocess_data(self):
        """预处理数据"""
        # 性别编码
        gender_map = {
            'Male': 0, 'Female': 1, 'MALE': 0, 'FEMALE': 1,
            'male': 0, 'female': 1, 'M': 0, 'F': 1
        }
        self.slide_data['gender_encoded'] = (
            self.slide_data['gender']
            .map(gender_map)
            .fillna(0)
            .astype(int)
        )
        
        # 年龄处理
        self.slide_data['age'] = pd.to_numeric(
            self.slide_data['age'], 
            errors='coerce'
        ).fillna(-1)
        
        # 解析slide_id列表
        if 'slide_id' in self.slide_data.columns:
            self.slide_data['slide_id_list'] = self.slide_data['slide_id'].apply(
                lambda x: ast.literal_eval(x.strip()) if isinstance(x, str) else [x]
            )
        
        # 确保必要的列存在
        required_cols = ['case_id', self.label_col, 'survival_months', 'censorship']
        missing_cols = [col for col in required_cols if col not in self.slide_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def _build_patient_mapping(self):
        """构建患者到切片的映射"""
        self.patient_to_slides = {}
        
        for idx, row in self.slide_data.iterrows():
            case_id = row['case_id']
            slide_ids = row.get('slide_id_list', [])
            
            if case_id not in self.patient_to_slides:
                self.patient_to_slides[case_id] = []
            
            self.patient_to_slides[case_id].extend(slide_ids)
        
        # 去重
        for case_id in self.patient_to_slides:
            self.patient_to_slides[case_id] = list(set(self.patient_to_slides[case_id]))
    
    def _prepare_patient_data(self):
        """准备患者级别的数据 (每个患者一条记录)"""
        # 按患者分组,取第一条记录作为患者数据
        patient_df = self.slide_data.groupby('case_id').first().reset_index()
        
        self.patient_data = {
            'case_id': patient_df['case_id'].values,
            'label': patient_df[self.label_col].values,
            'survival_months': patient_df['survival_months'].values,
            'censorship': patient_df['censorship'].values,
            'gender': patient_df['gender_encoded'].values,
            'age': patient_df['age'].values
        }
        
        self.num_patients = len(self.patient_data['case_id'])
        self.num_classes = len(np.unique(self.patient_data['label']))
    
    def _prepare_class_indices(self):
        """准备每个类别的患者索引 (用于类别平衡采样)"""
        self.patient_cls_ids = [[] for _ in range(self.num_classes)]
        
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(
                self.patient_data['label'] == i
            )[0]
    
    def summarize(self):
        """打印数据集统计信息"""
        print("=" * 60)
        print("Dataset Summary")
        print("=" * 60)
        print(f"Label column: {self.label_col}")
        print(f"Number of patients: {self.num_patients}")
        print(f"Number of classes: {self.num_classes}")
        print(f"H5 directory: {self.h5_dir}")
        
        print("\nPatient-level class distribution:")
        unique, counts = np.unique(self.patient_data['label'], return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"  Class {cls}: {count} patients ({count/self.num_patients*100:.1f}%)")
        
        print("\nSurvival statistics:")
        print(f"  Mean survival: {np.mean(self.patient_data['survival_months']):.2f} months")
        print(f"  Median survival: {np.median(self.patient_data['survival_months']):.2f} months")
        print(f"  Censorship rate: {np.mean(self.patient_data['censorship'])*100:.1f}%")
        
        print("\nSlides per patient:")
        slides_per_patient = [len(slides) for slides in self.patient_to_slides.values()]
        print(f"  Mean: {np.mean(slides_per_patient):.2f}")
        print(f"  Median: {np.median(slides_per_patient):.0f}")
        print(f"  Max: {np.max(slides_per_patient)}")
        print("=" * 60)
    
    def create_splits(self, n_splits=5, val_ratio=0.15, test_ratio=0.15, stratify=True):
        """
        创建K-fold交叉验证分割 (患者级别)
        
        Args:
            n_splits: fold数量
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            stratify: 是否按类别分层
        """
        self.n_splits = n_splits
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        n_patients = self.num_patients
        labels = self.patient_data['label']
        
        if stratify:
            # 分层K-fold
            skf = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.seed
            )
            self.splits = list(skf.split(np.zeros(n_patients), labels))
        else:
            # 普通K-fold
            from sklearn.model_selection import KFold
            kf = KFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.seed
            )
            self.splits = list(kf.split(np.zeros(n_patients)))
        
        print(f"\nCreated {n_splits}-fold cross-validation splits")
        print(f"  Validation ratio: {val_ratio}")
        print(f"  Test ratio: {test_ratio}")
        print(f"  Stratified: {stratify}")
    
    def set_split(self, fold=0):
        """
        设置当前使用的fold
        
        Args:
            fold: fold索引 (0 to n_splits-1)
        """
        if not hasattr(self, 'splits'):
            raise ValueError("Please call create_splits() first!")
        
        if fold >= self.n_splits:
            raise ValueError(f"fold must be < {self.n_splits}")
        
        train_val_idx, test_idx = self.splits[fold]
        
        # 从train_val中分出验证集
        n_val = int(len(train_val_idx) * self.val_ratio / (1 - self.test_ratio))
        
        np.random.seed(self.seed + fold)
        np.random.shuffle(train_val_idx)
        
        val_idx = train_val_idx[:n_val]
        train_idx = train_val_idx[n_val:]
        
        self.train_ids = train_idx
        self.val_ids = val_idx
        self.test_ids = test_idx
        
        print(f"\nSet fold {fold}:")
        print(f"  Train: {len(train_idx)} patients")
        print(f"  Val: {len(val_idx)} patients")
        print(f"  Test: {len(test_idx)} patients")
        
        # 打印每个split的类别分布
        for split_name, split_ids in [('Train', train_idx), ('Val', val_idx), ('Test', test_idx)]:
            split_labels = self.patient_data['label'][split_ids]
            unique, counts = np.unique(split_labels, return_counts=True)
            print(f"  {split_name} class distribution: {dict(zip(unique, counts))}")
        
        # 验证没有重叠
        assert len(np.intersect1d(train_idx, val_idx)) == 0
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        assert len(np.intersect1d(val_idx, test_idx)) == 0
    
    def get_split_dataset(self, split='train'):
        """
        获取指定split的数据集
        
        Args:
            split: 'train', 'val', 或 'test'
        
        Returns:
            PrognosisSplit对象
        """
        if split == 'train':
            patient_ids = self.train_ids
        elif split == 'val':
            patient_ids = self.val_ids
        elif split == 'test':
            patient_ids = self.test_ids
        else:
            raise ValueError(f"Invalid split: {split}")
        
        return PrognosisSplit(
            parent_dataset=self,
            patient_indices=patient_ids,
            split_name=split
        )
    
    def get_patient_samples(self, class_id, n_samples, replace=False):
        """
        从指定类别中采样患者 (用于类别平衡)
        
        Args:
            class_id: 类别ID
            n_samples: 采样数量
            replace: 是否有放回采样
        
        Returns:
            患者索引数组
        """
        if class_id >= self.num_classes:
            raise ValueError(f"class_id must be < {self.num_classes}")
        
        available_ids = self.patient_cls_ids[class_id]
        
        if n_samples > len(available_ids) and not replace:
            print(f"Warning: Requested {n_samples} samples but only {len(available_ids)} available")
            n_samples = len(available_ids)
        
        indices = np.random.choice(
            available_ids,
            size=n_samples,
            replace=replace
        )
        
        return indices

    def load_external_test(self, csv_path, h5_dir):
        """
        加载外部测试集
        
        Args:
            csv_path: 外部测试集CSV路径
            h5_dir: 外部测试集H5目录
        
        Returns:
            external_test_dataset: PrognosisSplit对象
        """
        print("\n" + "="*60)
        print("Loading External Test Set")
        print("="*60)
        
        # 创建一个新的数据集实例
        external_dataset = PrognosisDataset(
            csv_path=csv_path,
            h5_dir=h5_dir,
            label_col=self.label_col,
            shuffle=False,
            seed=self.seed,
            use_cache=self.use_cache,
            print_info=True
        )
        
        # 获取所有患者索引
        all_indices = np.arange(external_dataset.num_patients)
        
        # 创建一个 PrognosisSplit 对象
        external_test = PrognosisSplit(
            parent_dataset=external_dataset,
            patient_indices=all_indices,
            split_name='external_test'
        )
        
        print(f"External test set loaded: {len(external_test)} patients")
        print("="*60)
        
        return external_test
    
    def save_split(self, filename):
        """保存当前的数据分割"""
        split_data = {
            'train': self.patient_data['case_id'][self.train_ids].tolist(),
            'val': self.patient_data['case_id'][self.val_ids].tolist(),
            'test': self.patient_data['case_id'][self.test_ids].tolist()
        }
        
        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in split_data.items()]))
        df.to_csv(filename, index=False)
        print(f"Split saved to {filename}")
    
    def load_split(self, filename):
        """从文件加载数据分割"""
        df = pd.read_csv(filename)
        
        # 将case_id转换为索引
        case_id_to_idx = {
            case_id: idx 
            for idx, case_id in enumerate(self.patient_data['case_id'])
        }
        
        self.train_ids = np.array([
            case_id_to_idx[cid] 
            for cid in df['train'].dropna()
            if cid in case_id_to_idx
        ])
        self.val_ids = np.array([
            case_id_to_idx[cid] 
            for cid in df['val'].dropna()
            if cid in case_id_to_idx
        ])
        self.test_ids = np.array([
            case_id_to_idx[cid] 
            for cid in df['test'].dropna()
            if cid in case_id_to_idx
        ])
        
        print(f"Split loaded from {filename}")
        print(f"  Train: {len(self.train_ids)} patients")
        print(f"  Val: {len(self.val_ids)} patients")
        print(f"  Test: {len(self.test_ids)} patients")
    
    def __len__(self):
        """返回患者数量"""
        return self.num_patients
    
    def __getitem__(self, patient_idx):
        """
        获取一个患者的数据
        
        Returns:
            dict: {
                'case_id': 患者ID,
                'gender': 性别,
                'age': 年龄,
                'label': 标签,
                'survival_time': 生存时间,
                'censorship': 删失状态,
                'features': 特征张量,
                'coords': 坐标张量,
                'num_patches': patch数量
            }
        """
        # 获取患者信息
        case_id = self.patient_data['case_id'][patient_idx]
        gender = int(self.patient_data['gender'][patient_idx])
        age = float(self.patient_data['age'][patient_idx])
        label = int(self.patient_data['label'][patient_idx])
        survival_time = float(self.patient_data['survival_months'][patient_idx])
        censorship = int(self.patient_data['censorship'][patient_idx])
        
        # 获取该患者的所有切片
        slide_ids = self.patient_to_slides.get(case_id, [])
        
        if not slide_ids:
            raise ValueError(f"No slides found for patient {case_id}")
        
        # 加载并拼接所有切片的特征
        path_features = []
        path_coords = []
        
        for slide_id in slide_ids:
            # 构建H5文件路径
            h5_id = slide_id.strip().replace('.pt', '.h5')
            h5_path = os.path.join(self.h5_dir, h5_id)
            
            # 检查缓存
            cache_key = h5_path
            if self.use_cache and cache_key in self.data_cache:
                features, coords = self.data_cache[cache_key]
            else:
                # 加载H5文件
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
                        
                        features = torch.tensor(features, dtype=torch.float32)
                        coords = torch.tensor(coords, dtype=torch.float32)
                        
                        # 缓存
                        if self.use_cache:
                            self.data_cache[cache_key] = (features, coords)
                
                except Exception as e:
                    print(f'Error loading {h5_path}: {e}')
                    continue
            
            path_features.append(features)
            path_coords.append(coords)
        
        if not path_features:
            raise ValueError(f"No valid features loaded for patient {case_id}")
        
        # 拼接所有切片
        features = torch.cat(path_features, dim=0)
        coords = torch.cat(path_coords, dim=0)
        num_patches = features.shape[0]
        
        return {
            'case_id': case_id,
            'gender': gender,
            'age': age,
            'label': label,
            'survival_time': survival_time,
            'censorship': censorship,
            'features': features,
            'coords': coords,
            'num_patches': num_patches
        }


class PrognosisSplit(Dataset):
    """
    数据集的一个split (train/val/test)
    """
    def __init__(self, parent_dataset, patient_indices, split_name='train'):
        self.parent = parent_dataset
        self.patient_indices = patient_indices
        self.split_name = split_name
        self.use_cache = parent_dataset.use_cache
        self.h5_dir = parent_dataset.h5_dir
        
        # 准备该split的类别索引
        self._prepare_class_indices()
    
    def _prepare_class_indices(self):
        """准备该split中每个类别的样本索引"""
        labels = [
            self.parent.patient_data['label'][idx] 
            for idx in self.patient_indices
        ]
        
        self.num_classes = self.parent.num_classes
        self.cls_ids = [[] for _ in range(self.num_classes)]
        
        for local_idx, label in enumerate(labels):
            self.cls_ids[label].append(local_idx)
    
    def __len__(self):
        return len(self.patient_indices)
    
    def __getitem__(self, local_idx):
        """
        Args:
            local_idx: 在该split中的索引
        """
        # 转换为全局索引
        global_idx = self.patient_indices[local_idx]
        return self.parent[global_idx]
    
    def get_class_samples(self, class_id, n_samples, replace=False):
        """从该split的指定类别中采样"""
        if class_id >= self.num_classes:
            raise ValueError(f"class_id must be < {self.num_classes}")
        
        available_ids = self.cls_ids[class_id]
        
        if n_samples > len(available_ids) and not replace:
            n_samples = len(available_ids)
        
        return np.random.choice(available_ids, size=n_samples, replace=replace)


# ============= 自定义Collate函数 =============
def custom_collate_fn(batch):
    """
    自定义collate函数,用于处理变长的patch序列
    
    Args:
        batch: list of dict, 每个dict包含:
            - case_id: 患者ID
            - gender: 性别
            - age: 年龄
            - label: 标签
            - survival_time: 生存时间
            - censorship: 删失状态
            - features: [num_patches, feature_dim]
            - coords: [num_patches, 2]
            - num_patches: patch数量
    
    Returns:
        tuple: (patient_list, gender_tensor, age_tensor, label_tensor, 
                sur_time_tensor, censor_tensor, path_features, coords_tensor, 
                num_patch_tensor, mask_tensor)
    """
    # 过滤掉None
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    # 提取数据
    patient_list = [item['case_id'] for item in batch]
    gender_list = [item['gender'] for item in batch]
    age_list = [item['age'] for item in batch]
    label_list = [item['label'] for item in batch]
    sur_time_list = [item['survival_time'] for item in batch]
    censor_list = [item['censorship'] for item in batch]
    
    # 找到最大patch数
    max_patch_count = max(item['num_patches'] for item in batch)
    
    path_features_list = []
    coords_list = []
    mask_list = []
    num_patch_list = []
    
    for item in batch:
        features = item['features']  # [num_patches, feature_dim]
        coords = item['coords']      # [num_patches, 2]
        num_patches = item['num_patches']
        
        # Padding features
        if features.size(0) < max_patch_count:
            padding = torch.zeros(
                max_patch_count - features.size(0), 
                features.size(1),
                dtype=features.dtype
            )
            features = torch.cat((features, padding), dim=0)
        
        # Padding coords
        if coords.size(0) < max_patch_count:
            coords_padding = torch.zeros(
                max_patch_count - coords.size(0), 
                coords.size(1),
                dtype=coords.dtype
            )
            coords = torch.cat((coords, coords_padding), dim=0)
        
        # 创建mask (1表示真实patch, 0表示padding)
        mask = torch.ones(max_patch_count, dtype=torch.float)
        mask[num_patches:] = 0
        
        path_features_list.append(features)
        coords_list.append(coords)
        mask_list.append(mask)
        num_patch_list.append(num_patches)
    
    # 转换为tensor
    gender_tensor = torch.tensor(gender_list, dtype=torch.long)
    age_tensor = torch.tensor(age_list, dtype=torch.float)
    label_tensor = torch.tensor(label_list, dtype=torch.long)
    sur_time_tensor = torch.tensor(sur_time_list, dtype=torch.float)
    censor_tensor = torch.tensor(censor_list, dtype=torch.float)
    
    # Stack成batch
    path_features = torch.stack(path_features_list, dim=0)  # [batch, max_patches, feat_dim]
    coords_tensor = torch.stack(coords_list, dim=0)         # [batch, max_patches, 2]
    mask_tensor = torch.stack(mask_list, dim=0)             # [batch, max_patches]
    num_patch_tensor = torch.tensor(num_patch_list, dtype=torch.long)  # [batch]
    
    return (
        patient_list,      # list of str
        gender_tensor,     # [batch]
        age_tensor,        # [batch]
        label_tensor,      # [batch]
        sur_time_tensor,   # [batch]
        censor_tensor,     # [batch]
        path_features,     # [batch, max_patches, feat_dim]
        coords_tensor,     # [batch, max_patches, 2]
        num_patch_tensor,  # [batch]
        mask_tensor        # [batch, max_patches]
    )


# ============= 使用示例 =============
if __name__ == '__main__':
    print("="*60)
    print("Testing PrognosisDataset")
    print("="*60)
    
    # 1. 创建数据集
    dataset = PrognosisDataset(
        csv_path='/home/stat-jijianxin/PFMs/Survival_code/csv_file/tcga_survival_matched.csv',
        h5_dir='/home/stat-jijianxin/PFMs/TRIDENT/tcga_filtered/20x_512px_0px_overlap/features_conch_v15',
        use_cache=True,
        print_info=True
    )
    
    # 2. 创建5-fold交叉验证
    dataset.create_splits(n_splits=5, val_ratio=0.15, test_ratio=0.15)
    
    # 3. 设置使用第0个fold
    dataset.set_split(fold=0)
    
    # 4. 获取train/val/test数据集
    train_dataset = dataset.get_split_dataset('train')
    val_dataset = dataset.get_split_dataset('val')
    test_dataset = dataset.get_split_dataset('test')
    
    print(f"\nTrain dataset: {len(train_dataset)} patients")
    print(f"Val dataset: {len(val_dataset)} patients")
    print(f"Test dataset: {len(test_dataset)} patients")
    
    # 5. 测试加载单个样本
    print("\n" + "="*60)
    print("Testing single sample loading")
    print("="*60)
    sample = train_dataset[0]
    print(f"Patient: {sample['case_id']}")
    print(f"Gender: {sample['gender']}, Age: {sample['age']}")
    print(f"Label: {sample['label']}, Survival: {sample['survival_time']} months, Censored: {sample['censorship']}")
    print(f"Features shape: {sample['features'].shape}")
    print(f"Coords shape: {sample['coords'].shape}")
    print(f"Num patches: {sample['num_patches']}")
    
    # 6. 测试DataLoader + custom_collate_fn
    print("\n" + "="*60)
    print("Testing DataLoader with custom_collate_fn")
    print("="*60)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # 测试时用0,实际训练可以增加
        collate_fn=custom_collate_fn,
        pin_memory=False
    )
    import ipdb;ipdb.set_trace()
    # 加载一个batch测试
    for batch_idx, batch_data in enumerate(train_loader):
        if batch_data is None:
            print("Warning: Got None batch")
            continue
            
        (patient_list, gender, age, label, sur_time, censor, 
         features, coords, num_patches, mask) = batch_data
        
        print(f"\nBatch {batch_idx}:")
        print(f"  Batch size: {len(patient_list)}")
        print(f"  Patient IDs: {patient_list}")
        print(f"  Features shape: {features.shape}")      # [batch, max_patches, feat_dim]
        print(f"  Coords shape: {coords.shape}")          # [batch, max_patches, 2]
        print(f"  Mask shape: {mask.shape}")              # [batch, max_patches]
        print(f"  Num patches: {num_patches.tolist()}")   # [batch]
        print(f"  Labels: {label.tolist()}")              # [batch]
        print(f"  Survival times: {sur_time.tolist()}")   # [batch]
        print(f"  Censorship: {censor.tolist()}")         # [batch]
        print(f"  Gender: {gender.tolist()}")             # [batch]
        print(f"  Age: {age.tolist()}")                   # [batch]
        
        # 只测试第一个batch
        break
    
    # 7. 类别平衡采样测试
    print("\n" + "="*60)
    print("Testing class-balanced sampling")
    print("="*60)
    for cls in range(dataset.num_classes):
        samples = train_dataset.get_class_samples(cls, n_samples=5)
        print(f"Class {cls}: sampled {len(samples)} patients")
    
    # 8. 保存split
    print("\n" + "="*60)
    print("Saving split")
    print("="*60)
    dataset.save_split('split_fold0.csv')
    
    print("\n" + "="*60)
    print("All tests passed! ✅")
    print("="*60)
