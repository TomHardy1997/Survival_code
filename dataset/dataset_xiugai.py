import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np
import h5py
import ast
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import warnings


class PrognosisDataset(Dataset):
    """
    生存预后数据集
    
    🔥 新增功能:
    1. 年龄自动标准化
    2. 保存标准化参数用于测试集
    """
    def __init__(self, 
                 csv_path,
                 h5_base_dir,
                 feature_models='uni_v1',
                 label_col='disc_label',
                 shuffle=False,
                 seed=42,
                 use_cache=True,
                 max_cache_size=1000,
                 normalize_age=True,  # 🔥 新增: 是否标准化年龄
                 age_scaler=None,     # 🔥 新增: 外部提供的scaler
                 print_info=True):
        """
        Args:
            normalize_age: 是否标准化年龄 (默认True)
            age_scaler: 外部提供的StandardScaler (用于测试集)
        """
        self.seed = seed
        self.use_cache = use_cache
        self.max_cache_size = max_cache_size
        self.print_info = print_info
        self.label_col = label_col
        self.normalize_age = normalize_age
        
        # 处理特征模型参数
        if isinstance(feature_models, str):
            self.feature_models = [feature_models]
        elif isinstance(feature_models, list):
            self.feature_models = feature_models
        else:
            raise ValueError("feature_models must be str or list of str")
        
        # 构建H5目录列表
        self.h5_base_dir = h5_base_dir
        self.h5_dirs = []
        for model in self.feature_models:
            h5_dir = os.path.join(h5_base_dir, f'features_{model}')
            if not os.path.exists(h5_dir):
                raise ValueError(f"H5 directory not found: {h5_dir}")
            self.h5_dirs.append(h5_dir)
        
        # 数据缓存
        self.data_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # 加载CSV数据
        self.slide_data = pd.read_csv(csv_path)
        
        # 预处理数据
        self._preprocess_data(age_scaler)
        
        # 构建患者-切片映射
        self._build_patient_mapping()
        
        # 准备患者级别数据
        self._prepare_patient_data()
        
        # 准备类别索引
        self._prepare_class_indices()
        
        # 验证特征维度
        self._validate_feature_dims()
        
        # 初始化分割
        self.train_ids = []
        self.val_ids = []
        self.test_ids = []
        
        if shuffle:
            np.random.seed(seed)
            patient_indices = np.arange(self.num_patients)
            np.random.shuffle(patient_indices)
            for key in self.patient_data:
                self.patient_data[key] = self.patient_data[key][patient_indices]
        
        if print_info:
            self.summarize()
    
    def _preprocess_data(self, external_age_scaler=None):
        """
        预处理数据
        
        🔥 改进: 添加年龄标准化
        """
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
        
        # 🔥 年龄处理 - 改进版
        self.slide_data['age'] = pd.to_numeric(
            self.slide_data['age'], 
            errors='coerce'
        )
        
        # 计算年龄均值用于填充缺失值
        age_mean = self.slide_data['age'].mean()
        if pd.isna(age_mean):
            age_mean = 60.0
            warnings.warn("All ages are missing, using default value 60.0")
        
        self.slide_data['age'] = self.slide_data['age'].fillna(age_mean)
        
        # 🔥 年龄标准化
        if self.normalize_age:
            if external_age_scaler is not None:
                # 使用外部提供的scaler (用于测试集)
                self.age_scaler = external_age_scaler
                self.slide_data['age_normalized'] = self.age_scaler.transform(
                    self.slide_data[['age']]
                ).flatten()
                print(f"✓ Using external age scaler")
            else:
                # 训练新的scaler
                self.age_scaler = StandardScaler()
                self.slide_data['age_normalized'] = self.age_scaler.fit_transform(
                    self.slide_data[['age']]
                ).flatten()
                print(f"✓ Age normalized: mean={self.age_scaler.mean_[0]:.2f}, "
                      f"std={self.age_scaler.scale_[0]:.2f}")
            
            # 使用标准化后的年龄
            self.slide_data['age'] = self.slide_data['age_normalized']
        else:
            self.age_scaler = None
            print("⚠ Age normalization disabled")
        
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
        """准备患者级别的数据"""
        patient_df = self.slide_data.groupby('case_id').first().reset_index()
        
        self.patient_data = {
            'case_id': patient_df['case_id'].values,
            'label': patient_df[self.label_col].values,
            'survival_months': patient_df['survival_months'].values,
            'censorship': patient_df['censorship'].values,
            'gender': patient_df['gender_encoded'].values,
            'age': patient_df['age'].values  # 已经是标准化后的值
        }
        
        self.num_patients = len(self.patient_data['case_id'])
        self.num_classes = len(np.unique(self.patient_data['label']))
    
    def _prepare_class_indices(self):
        """准备每个类别的患者索引"""
        self.patient_cls_ids = [[] for _ in range(self.num_classes)]
        
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(
                self.patient_data['label'] == i
            )[0]
    
    def _validate_feature_dims(self):
        """验证所有H5文件的特征维度一致性"""
        print("\n" + "="*60)
        print("Validating Feature Dimensions...")
        print("="*60)
        
        self.feature_dims = []
        
        for model, h5_dir in zip(self.feature_models, self.h5_dirs):
            h5_files = [f for f in os.listdir(h5_dir) if f.endswith('.h5')]
            if not h5_files:
                raise ValueError(f"No H5 files found in {h5_dir}")
            
            sample_file = h5_files[0]
            sample_path = os.path.join(h5_dir, sample_file)
            
            try:
                with h5py.File(sample_path, 'r') as f:
                    feat_dim = f['features'].shape[1]
                    self.feature_dims.append(feat_dim)
                    print(f"  Model '{model}': feature_dim = {feat_dim}")
            except Exception as e:
                raise RuntimeError(f"Error reading {sample_path}: {e}")
        
        self.total_feature_dim = sum(self.feature_dims)
        print(f"\nTotal concatenated feature dimension: {self.total_feature_dim}")
        print("="*60)
    
    def get_class_weights(self):
        """计算类别权重"""
        unique, counts = np.unique(self.patient_data['label'], return_counts=True)
        weights = 1.0 / counts
        weights = weights / weights.sum() * len(weights)
        return torch.FloatTensor(weights)
    
    def get_cache_stats(self):
        """获取缓存统计信息"""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_size': len(self.data_cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate
        }
    
    def get_age_scaler(self):
        """
        🔥 新增: 获取年龄标准化器 (用于测试集)
        
        Returns:
            StandardScaler or None
        """
        return self.age_scaler
    
    def summarize(self):
        """打印数据集统计信息"""
        print("=" * 60)
        print("Dataset Summary")
        print("=" * 60)
        print(f"Label column: {self.label_col}")
        print(f"Number of patients: {self.num_patients}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Feature models: {', '.join(self.feature_models)}")
        print(f"H5 base directory: {self.h5_base_dir}")
        for i, (model, h5_dir) in enumerate(zip(self.feature_models, self.h5_dirs)):
            print(f"  Model {i+1}: {model} -> {h5_dir}")
        
        print("\nPatient-level class distribution:")
        unique, counts = np.unique(self.patient_data['label'], return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"  Class {cls}: {count} patients ({count/self.num_patients*100:.1f}%)")
        
        class_weights = self.get_class_weights()
        print("\nClass weights (for balanced loss):")
        for cls, weight in enumerate(class_weights):
            print(f"  Class {cls}: {weight:.4f}")
        
        print("\nSurvival statistics:")
        print(f"  Mean survival: {np.mean(self.patient_data['survival_months']):.2f} months")
        print(f"  Median survival: {np.median(self.patient_data['survival_months']):.2f} months")
        print(f"  Censorship rate: {np.mean(self.patient_data['censorship'])*100:.1f}%")
        
        # 🔥 改进: 显示年龄统计
        print("\nAge statistics:")
        if self.normalize_age:
            print(f"  ✓ Normalized (mean=0, std=1)")
            print(f"  Range: [{np.min(self.patient_data['age']):.2f}, "
                  f"{np.max(self.patient_data['age']):.2f}]")
            if self.age_scaler is not None:
                print(f"  Original mean: {self.age_scaler.mean_[0]:.2f}")
                print(f"  Original std: {self.age_scaler.scale_[0]:.2f}")
        else:
            print(f"  Mean age: {np.mean(self.patient_data['age']):.2f}")
            print(f"  Median age: {np.median(self.patient_data['age']):.2f}")
            print(f"  Age range: [{np.min(self.patient_data['age']):.0f}, "
                  f"{np.max(self.patient_data['age']):.0f}]")
        
        print("\nSlides per patient:")
        slides_per_patient = [len(slides) for slides in self.patient_to_slides.values()]
        print(f"  Mean: {np.mean(slides_per_patient):.2f}")
        print(f"  Median: {np.median(slides_per_patient):.0f}")
        print(f"  Max: {np.max(slides_per_patient)}")
        print("=" * 60)
    
    def create_splits(self, n_splits=5, val_ratio=0.15, test_ratio=0.15, stratify=True):
        """创建K-fold交叉验证分割"""
        self.n_splits = n_splits
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        n_patients = self.num_patients
        labels = self.patient_data['label']
        
        if stratify:
            skf = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.seed
            )
            self.splits = list(skf.split(np.zeros(n_patients), labels))
        else:
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
        """设置当前使用的fold"""
        if not hasattr(self, 'splits'):
            raise ValueError("Please call create_splits() first!")
        
        if fold >= self.n_splits:
            raise ValueError(f"fold must be < {self.n_splits}")
        
        train_val_idx, test_idx = self.splits[fold]
        
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
        
        for split_name, split_ids in [('Train', train_idx), ('Val', val_idx), ('Test', test_idx)]:
            split_labels = self.patient_data['label'][split_ids]
            unique, counts = np.unique(split_labels, return_counts=True)
            print(f"  {split_name} class distribution: {dict(zip(unique, counts))}")
        
        assert len(np.intersect1d(train_idx, val_idx)) == 0
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        assert len(np.intersect1d(val_idx, test_idx)) == 0
    
    def get_split_dataset(self, split='train'):
        """获取指定split的数据集"""
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
        """从指定类别中采样患者"""
        if class_id >= self.num_classes:
            raise ValueError(f"class_id must be < {self.num_classes}")
        
        available_ids = self.patient_cls_ids[class_id]
        
        if n_samples > len(available_ids) and not replace:
            warnings.warn(f"Requested {n_samples} samples but only {len(available_ids)} available")
            n_samples = len(available_ids)
        
        indices = np.random.choice(
            available_ids,
            size=n_samples,
            replace=replace
        )
        
        return indices

    def load_external_test(self, csv_path, h5_base_dir=None, feature_models=None):
        """
        加载外部测试集
        
        🔥 改进: 自动传递age_scaler
        """
        print("\n" + "="*60)
        print("Loading External Test Set")
        print("="*60)
        
        if h5_base_dir is None:
            h5_base_dir = self.h5_base_dir
        if feature_models is None:
            feature_models = self.feature_models
        
        # 🔥 关键: 传递age_scaler到外部测试集
        external_dataset = PrognosisDataset(
            csv_path=csv_path,
            h5_base_dir=h5_base_dir,
            feature_models=feature_models,
            label_col=self.label_col,
            shuffle=False,
            seed=self.seed,
            use_cache=self.use_cache,
            max_cache_size=self.max_cache_size,
            normalize_age=self.normalize_age,
            age_scaler=self.age_scaler,  # 🔥 使用训练集的scaler
            print_info=True
        )
        
        all_indices = np.arange(external_dataset.num_patients)
        
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
    
    def _load_features_from_h5(self, h5_path):
        """从单个H5文件加载特征"""
        if self.use_cache and h5_path in self.data_cache:
            self._cache_hits += 1
            return self.data_cache[h5_path]
        
        self._cache_misses += 1
        
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f'File not found: {h5_path}')
        
        try:
            with h5py.File(h5_path, 'r') as f:
                features = f['features'][:]
                coords = f['coords'][:]
                
                if features.shape[0] != coords.shape[0]:
                    raise ValueError(f'Shape mismatch in {h5_path}')
                
                features = torch.tensor(features, dtype=torch.float32)
                coords = torch.tensor(coords, dtype=torch.float32)
                
                if self.use_cache:
                    if len(self.data_cache) >= self.max_cache_size:
                        first_key = next(iter(self.data_cache))
                        del self.data_cache[first_key]
                    
                    self.data_cache[h5_path] = (features, coords)
                
                return features, coords
        
        except Exception as e:
            raise RuntimeError(f'Error loading {h5_path}: {e}')
    
    def __len__(self):
        """返回患者数量"""
        return self.num_patients
    
    def __getitem__(self, patient_idx):
        """获取一个患者的数据"""
        case_id = self.patient_data['case_id'][patient_idx]
        gender = int(self.patient_data['gender'][patient_idx])
        age = float(self.patient_data['age'][patient_idx])  # 已经是标准化后的值
        label = int(self.patient_data['label'][patient_idx])
        survival_time = float(self.patient_data['survival_months'][patient_idx])
        censorship = int(self.patient_data['censorship'][patient_idx])
        
        slide_ids = self.patient_to_slides.get(case_id, [])
        
        if not slide_ids:
            raise ValueError(f"No slides found for patient {case_id}")
        
        all_slide_features = []
        all_slide_coords = []
        
        for slide_id in slide_ids:
            h5_id = slide_id.strip().replace('.pt', '.h5')
            
            slide_features_list = []
            slide_coords = None
            
            for h5_dir in self.h5_dirs:
                h5_path = os.path.join(h5_dir, h5_id)
                
                try:
                    features, coords = self._load_features_from_h5(h5_path)
                    slide_features_list.append(features)
                    
                    if slide_coords is None:
                        slide_coords = coords
                    
                except Exception as e:
                    warnings.warn(f'Error loading {h5_path}: {e}')
                    continue
            
            if not slide_features_list:
                warnings.warn(f'No valid features loaded for slide {slide_id}')
                continue
            
            if len(slide_features_list) > 1:
                patch_counts = [f.shape[0] for f in slide_features_list]
                if len(set(patch_counts)) > 1:
                    raise ValueError(
                        f"Patch count mismatch for slide {slide_id}: {patch_counts}"
                    )
            
            if len(slide_features_list) > 1:
                slide_features = torch.cat(slide_features_list, dim=1)
            else:
                slide_features = slide_features_list[0]
            
            all_slide_features.append(slide_features)
            all_slide_coords.append(slide_coords)
        
        if not all_slide_features:
            raise ValueError(f"No valid features loaded for patient {case_id}")
        
        features = torch.cat(all_slide_features, dim=0)
        coords = torch.cat(all_slide_coords, dim=0)
        num_patches = features.shape[0]
        
        return {
            'case_id': case_id,
            'gender': gender,
            'age': age,  # 标准化后的年龄
            'label': label,
            'survival_time': survival_time,
            'censorship': censorship,
            'features': features,
            'coords': coords,
            'num_patches': num_patches
        }


class PrognosisSplit(Dataset):
    """数据集的一个split"""
    def __init__(self, parent_dataset, patient_indices, split_name='train'):
        self.parent = parent_dataset
        self.patient_indices = patient_indices
        self.split_name = split_name
        self.use_cache = parent_dataset.use_cache
        self.h5_dirs = parent_dataset.h5_dirs
        
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


def custom_collate_fn(batch):
    """自定义collate函数"""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    patient_list = [item['case_id'] for item in batch]
    gender_list = [item['gender'] for item in batch]
    age_list = [item['age'] for item in batch]
    label_list = [item['label'] for item in batch]
    sur_time_list = [item['survival_time'] for item in batch]
    censor_list = [item['censorship'] for item in batch]
    
    max_patch_count = max(item['num_patches'] for item in batch)
    
    path_features_list = []
    coords_list = []
    mask_list = []
    num_patch_list = []
    
    for item in batch:
        features = item['features']
        coords = item['coords']
        num_patches = item['num_patches']
        
        if features.size(0) < max_patch_count:
            padding = torch.zeros(
                max_patch_count - features.size(0), 
                features.size(1),
                dtype=features.dtype
            )
            features = torch.cat((features, padding), dim=0)
        
        if coords.size(0) < max_patch_count:
            coords_padding = torch.zeros(
                max_patch_count - coords.size(0), 
                coords.size(1),
                dtype=coords.dtype
            )
            coords = torch.cat((coords, coords_padding), dim=0)
        
        mask = torch.ones(max_patch_count, dtype=torch.float)
        mask[num_patches:] = 0
        
        path_features_list.append(features)
        coords_list.append(coords)
        mask_list.append(mask)
        num_patch_list.append(num_patches)
    
    gender_tensor = torch.tensor(gender_list, dtype=torch.long)
    age_tensor = torch.tensor(age_list, dtype=torch.float)
    label_tensor = torch.tensor(label_list, dtype=torch.long)
    sur_time_tensor = torch.tensor(sur_time_list, dtype=torch.float)
    censor_tensor = torch.tensor(censor_list, dtype=torch.float)
    
    path_features = torch.stack(path_features_list, dim=0)
    coords_tensor = torch.stack(coords_list, dim=0)
    mask_tensor = torch.stack(mask_list, dim=0)
    num_patch_tensor = torch.tensor(num_patch_list, dtype=torch.long)
    
    return {
        'case_id': patient_list,
        'gender': gender_tensor,
        'age': age_tensor,
        'label': label_tensor,
        'survival_time': sur_time_tensor,
        'censorship': censor_tensor,
        'features': path_features,
        'coords': coords_tensor,
        'num_patches': num_patch_tensor,
        'mask': mask_tensor
    }


if __name__ == '__main__':
    print("="*60)
    print("Testing PrognosisDataset with Age Normalization")
    print("="*60)
    
    # 🔥 测试1: 标准化年龄
    print("\n[Test 1] With age normalization")
    dataset1 = PrognosisDataset(
        csv_path='/home/stat-jijianxin/PFMs/Survival_code/csv_file/tcga_survival_matched.csv',
        h5_base_dir='/home/stat-jijianxin/PFMs/TRIDENT/tcga_filtered/20x_512px_0px_overlap',
        feature_models='uni_v1',
        normalize_age=True  # 开启标准化
    )
    
    # 🔥 测试2: 不标准化年龄
    print("\n[Test 2] Without age normalization")
    dataset2 = PrognosisDataset(
        csv_path='/home/stat-jijianxin/PFMs/Survival_code/csv_file/tcga_survival_matched.csv',
        h5_base_dir='/home/stat-jijianxin/PFMs/TRIDENT/tcga_filtered/20x_512px_0px_overlap',
        feature_models='uni_v1',
        normalize_age=False  # 关闭标准化
    )
    
    # 🔥 测试3: 外部测试集使用训练集的scaler
    print("\n[Test 3] External test with training scaler")
    external_test = dataset1.load_external_test(
        csv_path='/home/stat-jijianxin/PFMs/Survival_code/csv_file/hmu_survival_with_slides.csv'
    )
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
    import ipdb;ipdb.set_trace()
