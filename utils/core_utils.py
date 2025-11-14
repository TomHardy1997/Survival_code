"""
生存分析训练框架 - 针对 PrognosisDataset 和 Mamba2MIL
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sksurv.metrics import concordance_index_censored
import pandas as pd
from tqdm import tqdm
import pickle
from argparse import Namespace
from utils.survival_loss_function import CoxSurvLoss, NLLSurvLoss



# ===================== 早停机制 =====================
class EarlyStopping:
    """基于C-Index的早停"""
    def __init__(self, warmup=5, patience=15, stop_epoch=20, verbose=False):
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_cindex = 0

    def __call__(self, epoch, val_cindex, model, ckpt_name='checkpoint.pt'):
        score = val_cindex

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_cindex, model, ckpt_name)
        elif score <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_cindex, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_cindex, model, ckpt_name):
        if self.verbose:
            print(f'C-Index increased ({self.best_cindex:.4f} --> {val_cindex:.4f}). Saving model...')
        torch.save(model.state_dict(), ckpt_name)
        self.best_cindex = val_cindex


# ===================== 数据加载器 =====================
def collate_fn(batch):
    """
    自定义collate函数,处理不同长度的特征
    """
    # 提取所有字段
    case_ids = [item['case_id'] for item in batch]
    genders = torch.tensor([item['gender'] for item in batch])
    ages = torch.tensor([item['age'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    survival_times = torch.tensor([item['survival_time'] for item in batch])
    censorships = torch.tensor([item['censorship'] for item in batch])
    
    # 特征和坐标 - 每个样本可能有不同数量的patches
    features_list = [item['features'] for item in batch]
    coords_list = [item['coords'] for item in batch]
    num_patches = [item['num_patches'] for item in batch]
    
    return {
        'case_id': case_ids,
        'gender': genders,
        'age': ages,
        'label': labels,
        'survival_time': survival_times,
        'censorship': censorships,
        'features': features_list,  # List of tensors
        'coords': coords_list,      # List of tensors
        'num_patches': num_patches
    }


def get_split_loader(split_dataset, batch_size=1, num_workers=4):
    """
    创建数据加载器
    注意: 对于MIL,通常batch_size=1,因为每个患者的patch数量不同
    """
    loader = DataLoader(
        split_dataset,
        batch_size=batch_size,
        shuffle=(split_dataset.split_name == 'train'),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    return loader


# ===================== 训练循环 =====================
def train_loop(epoch, model, loader, optimizer, loss_fn, device, gc=1):
    """
    训练一个epoch
    
    Args:
        epoch: 当前epoch
        model: 模型
        loader: 数据加载器
        optimizer: 优化器
        loss_fn: 损失函数
        device: 设备
        gc: 梯度累积步数
    """
    model.train()
    train_loss = 0.
    
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, batch in pbar:
        # 获取数据 (batch_size=1)
        features = batch['features'][0].to(device)  # [n_patches, in_dim]
        label = batch['label'].to(device)           # [1]
        event_time = batch['survival_time']         # [1]
        c = batch['censorship'].to(device)          # [1]
        
        # 前向传播
        hazards, S, Y_hat, _, _ = model(features)   # features: [n_patches, in_dim]
        
        # 计算损失
        loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)
        loss_value = loss.item()
        
        # 计算风险分数
        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores.append(risk[0])
        all_censorships.append(c.item())
        all_event_times.append(event_time.item())
        
        train_loss += loss_value
        
        # 反向传播 (梯度累积)
        loss = loss / gc
        loss.backward()
        
        if (batch_idx + 1) % gc == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # 更新进度条
        pbar.set_postfix({'loss': f'{loss_value:.4f}'})
    
    # 最后一步
    if len(loader) % gc != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    # 计算平均损失
    train_loss /= len(loader)
    
    # 计算C-Index
    all_risk_scores = np.array(all_risk_scores)
    all_censorships = np.array(all_censorships)
    all_event_times = np.array(all_event_times)
    
    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool),
        all_event_times,
        all_risk_scores,
        tied_tol=1e-08
    )[0]
    
    print(f'Epoch {epoch}: train_loss={train_loss:.4f}, train_c_index={c_index:.4f}')
    
    return train_loss, c_index


# ===================== 验证循环 =====================
def validate(epoch, model, loader, loss_fn, device):
    """
    验证一个epoch
    """
    model.eval()
    val_loss = 0.
    
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch} [Val]')
    
    with torch.no_grad():
        for batch_idx, batch in pbar:
            features = batch['features'][0].to(device)
            label = batch['label'].to(device)
            event_time = batch['survival_time']
            c = batch['censorship'].to(device)
            
            # 前向传播
            hazards, S, Y_hat, _, _ = model(features)
            
            # 计算损失
            loss = loss_fn(hazards=hazards, S=S, Y=label, c=c, alpha=0)
            loss_value = loss.item()
            
            # 计算风险分数
            risk = -torch.sum(S, dim=1).cpu().numpy()
            all_risk_scores.append(risk[0])
            all_censorships.append(c.item())
            all_event_times.append(event_time.item())
            
            val_loss += loss_value
            
            pbar.set_postfix({'loss': f'{loss_value:.4f}'})
    
    val_loss /= len(loader)
    
    # 计算C-Index
    all_risk_scores = np.array(all_risk_scores)
    all_censorships = np.array(all_censorships)
    all_event_times = np.array(all_event_times)
    
    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool),
        all_event_times,
        all_risk_scores,
        tied_tol=1e-08
    )[0]
    
    print(f'Epoch {epoch}: val_loss={val_loss:.4f}, val_c_index={c_index:.4f}')
    
    return val_loss, c_index


# ===================== 测试函数 =====================
def test(model, loader, device):
    """
    在测试集上评估模型
    """
    model.eval()
    
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    patient_results = {}
    
    pbar = tqdm(enumerate(loader), total=len(loader), desc='Testing')
    
    with torch.no_grad():
        for batch_idx, batch in pbar:
            case_id = batch['case_id'][0]
            features = batch['features'][0].to(device)
            label = batch['label']
            event_time = batch['survival_time']
            c = batch['censorship']
            
            # 前向传播
            hazards, S, Y_hat, _, _ = model(features)
            
            # 计算风险分数
            risk = -torch.sum(S, dim=1).cpu().numpy()[0]
            
            all_risk_scores.append(risk)
            all_censorships.append(c.item())
            all_event_times.append(event_time.item())
            
            # 保存患者结果
            patient_results[case_id] = {
                'case_id': case_id,
                'risk': risk,
                'disc_label': label.item(),
                'survival': event_time.item(),
                'censorship': c.item(),
                'hazards': hazards.cpu().numpy(),
                'S': S.cpu().numpy()
            }
    
    # 计算C-Index
    all_risk_scores = np.array(all_risk_scores)
    all_censorships = np.array(all_censorships)
    all_event_times = np.array(all_event_times)
    
    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool),
        all_event_times,
        all_risk_scores,
        tied_tol=1e-08
    )[0]
    
    print(f'Test C-Index: {c_index:.4f}')
    
    return patient_results, c_index


# ===================== 主训练函数 =====================
def train_survival(args):
    """
    主训练函数 - 训练单个fold
    
    Args:
        args: 训练参数
    """
    print('\n' + '='*60)
    print(f'Training Fold {args.fold}')
    print('='*60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 创建结果目录
    fold_dir = os.path.join(args.results_dir, f'fold_{args.fold}')
    os.makedirs(fold_dir, exist_ok=True)
    
    # ========== 1. 加载数据集 ==========
    print('\n[1/7] Loading dataset...')
    from dataset.dataset_h5 import PrognosisDataset
    
    dataset = PrognosisDataset(
        csv_path=args.csv_path,
        h5_dir=args.h5_dir,
        label_col=args.label_col,
        use_cache=True,
        print_info=True
    )
    
    # 创建K-fold分割
    if not hasattr(dataset, 'splits'):
        dataset.create_splits(
            n_splits=args.k_fold,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            stratify=True
        )
    
    # 设置当前fold
    dataset.set_split(fold=args.fold)
    
    # 获取train/val/test数据集
    train_dataset = dataset.get_split_dataset('train')
    val_dataset = dataset.get_split_dataset('val')
    test_dataset = dataset.get_split_dataset('test')
    
    # ========== 加载外部测试集 ==========
    external_test_dataset = None
    if hasattr(args, 'external_csv_path') and args.external_csv_path:
        print('\n[1.5/7] Loading External Test Set...')
        external_test_dataset = dataset.load_external_test(
            csv_path=args.external_csv_path,
            h5_dir=args.external_h5_dir
        )
    
    print(f'\nDataset sizes:')
    print(f'  Train: {len(train_dataset)} patients')
    print(f'  Val: {len(val_dataset)} patients')
    print(f'  Test: {len(test_dataset)} patients')
    if external_test_dataset:
        print(f'  External Test: {len(external_test_dataset)} patients')
    
    # ========== 2. 创建数据加载器 ==========
    print('\n[2/7] Creating data loaders...')
    train_loader = get_split_loader(train_dataset, batch_size=1, num_workers=args.num_workers)
    val_loader = get_split_loader(val_dataset, batch_size=1, num_workers=args.num_workers)
    test_loader = get_split_loader(test_dataset, batch_size=1, num_workers=args.num_workers)
    
    # ========== 3. 初始化模型 ==========
    print('\n[3/7] Initializing model...')
    from models.Mamba2MIL import Mamba2MIL
    
    model = Mamba2MIL(
        in_dim=args.in_dim,
        n_classes=args.n_classes,
        dropout=args.dropout,
        act=args.act,
        survival=True,
        layer=args.mamba_layer
    )
    
    model.relocate()
    model = model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # ========== 4. 初始化优化器和损失函数 ==========
    print('\n[4/7] Initializing optimizer and loss function...')
    
    # 优化器
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f'Unknown optimizer: {args.optimizer}')
    
    # 损失函数
    if args.loss == 'cox':
        loss_fn = CoxSurvLoss()
    elif args.loss == 'nll':
        loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
    else:
        raise ValueError(f'Unknown loss: {args.loss}')
    
    print(f'Optimizer: {args.optimizer}')
    print(f'Loss function: {args.loss}')
    print(f'Learning rate: {args.lr}')
    print(f'Weight decay: {args.weight_decay}')
    
    # ========== 5. 训练循环 ==========
    print('\n[5/7] Training...')
    
    # 早停
    early_stopping = EarlyStopping(
        warmup=args.warmup,
        patience=args.patience,
        stop_epoch=args.stop_epoch,
        verbose=True
    )
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'train_cindex': [],
        'val_loss': [],
        'val_cindex': []
    }
    
    best_val_cindex = 0
    
    for epoch in range(args.max_epochs):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch+1}/{args.max_epochs}')
        print(f'{"="*60}')
        
        # 训练
        train_loss, train_cindex = train_loop(
            epoch=epoch,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            gc=args.gc
        )
        
        # 验证
        val_loss, val_cindex = validate(
            epoch=epoch,
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device
        )
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_cindex'].append(train_cindex)
        history['val_loss'].append(val_loss)
        history['val_cindex'].append(val_cindex)
        
        # 保存最佳模型
        if val_cindex > best_val_cindex:
            best_val_cindex = val_cindex
            torch.save(
                model.state_dict(),
                os.path.join(fold_dir, 'best_model.pt')
            )
            print(f'✓ Best model saved (val_cindex={val_cindex:.4f})')
        
        # 早停检查
        ckpt_path = os.path.join(fold_dir, 'checkpoint.pt')
        early_stopping(epoch, val_cindex, model, ckpt_name=ckpt_path)
        
        if early_stopping.early_stop:
            print(f'\nEarly stopping at epoch {epoch+1}')
            break
    
    # 保存训练历史
    with open(os.path.join(fold_dir, 'history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    
    # ========== 6. 测试 ==========
    print('\n[6/7] Testing...')
    
    # 加载最佳模型
    model.load_state_dict(torch.load(os.path.join(fold_dir, 'best_model.pt')))
    
    # 在验证集上评估
    print('\nEvaluating on validation set...')
    val_results, val_cindex = test(model, val_loader, device)
    
    # 在内部测试集上评估
    print('\nEvaluating on internal test set...')
    test_results, test_cindex = test(model, test_loader, device)
    
    # ========== 7. 在外部测试集上评估 ==========
    external_test_results = None
    external_test_cindex = None
    
    if external_test_dataset is not None:
        print('\n[7/7] Evaluating on External Test Set...')
        external_test_loader = get_split_loader(
            external_test_dataset,
            batch_size=1,
            num_workers=args.num_workers
        )
        external_test_results, external_test_cindex = test(
            model, 
            external_test_loader, 
            device
        )
        print(f'External Test C-Index: {external_test_cindex:.4f}')
    
    # 保存结果
    results = {
        'fold': args.fold,
        'val_cindex': val_cindex,
        'test_cindex': test_cindex,
        'val_results': val_results,
        'test_results': test_results,
        'history': history
    }
    
    # 添加外部测试集结果
    if external_test_results is not None:
        results['external_test_cindex'] = external_test_cindex
        results['external_test_results'] = external_test_results
    
    with open(os.path.join(fold_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # 保存CSV
    val_df = pd.DataFrame([v for v in val_results.values()])
    val_df.to_csv(os.path.join(fold_dir, 'val_results.csv'), index=False)
    
    test_df = pd.DataFrame([v for v in test_results.values()])
    test_df.to_csv(os.path.join(fold_dir, 'test_results.csv'), index=False)
    
    # 保存外部测试集结果
    if external_test_results is not None:
        external_df = pd.DataFrame([v for v in external_test_results.values()])
        external_df.to_csv(
            os.path.join(fold_dir, 'external_test_results.csv'), 
            index=False
        )
    
    print('\n' + '='*60)
    print('Training completed!')
    print('='*60)
    print(f'Validation C-Index: {val_cindex:.4f}')
    print(f'Internal Test C-Index: {test_cindex:.4f}')
    if external_test_cindex is not None:
        print(f'External Test C-Index: {external_test_cindex:.4f}')
    print(f'Results saved to: {fold_dir}')
    
    return results


# ===================== K-Fold交叉验证 =====================
def train_k_fold(args):
    """
    K-Fold交叉验证
    """
    print('\n' + '='*60)
    print(f'K-Fold Cross Validation (K={args.k_fold})')
    print('='*60)
    
    all_results = []
    
    for fold in range(args.k_fold):
        args.fold = fold
        results = train_survival(args)
        all_results.append(results)
    
    # 汇总结果
    val_cindices = [r['val_cindex'] for r in all_results]
    test_cindices = [r['test_cindex'] for r in all_results]
    
    # 汇总外部测试集结果
    external_cindices = []
    has_external = False
    for r in all_results:
        if 'external_test_cindex' in r:
            external_cindices.append(r['external_test_cindex'])
            has_external = True
    
    print('\n' + '='*60)
    print('K-Fold Cross Validation Results')
    print('='*60)
    
    for fold in range(args.k_fold):
        val_ci = val_cindices[fold]
        test_ci = test_cindices[fold]
        print(f'Fold {fold}: Val={val_ci:.4f}, Internal Test={test_ci:.4f}', end='')
        
        if has_external:
            ext_ci = external_cindices[fold]
            print(f', External Test={ext_ci:.4f}')
        else:
            print()
    
    print(f'\nMean Val C-Index: {np.mean(val_cindices):.4f} ± {np.std(val_cindices):.4f}')
    print(f'Mean Internal Test C-Index: {np.mean(test_cindices):.4f} ± {np.std(test_cindices):.4f}')
    
    if has_external:
        print(f'Mean External Test C-Index: {np.mean(external_cindices):.4f} ± {np.std(external_cindices):.4f}')
    
    # 保存汇总结果
    summary = {
        'val_cindices': val_cindices,
        'test_cindices': test_cindices,
        'mean_val_cindex': np.mean(val_cindices),
        'std_val_cindex': np.std(val_cindices),
        'mean_test_cindex': np.mean(test_cindices),
        'std_test_cindex': np.std(test_cindices),
        'all_results': all_results
    }
    
    if has_external:
        summary['external_cindices'] = external_cindices
        summary['mean_external_cindex'] = np.mean(external_cindices)
        summary['std_external_cindex'] = np.std(external_cindices)
    
    with open(os.path.join(args.results_dir, 'summary.pkl'), 'wb') as f:
        pickle.dump(summary, f)
    
    # 保存到CSV
    summary_data = {
        'fold': range(args.k_fold),
        'val_cindex': val_cindices,
        'test_cindex': test_cindices
    }
    
    if has_external:
        summary_data['external_test_cindex'] = external_cindices
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(args.results_dir, 'summary.csv'), index=False)
    
    print(f'\nSummary saved to: {args.results_dir}')
    
    return summary


# ===================== 主函数 =====================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Survival Analysis Training')
    
    # 数据参数
    parser.add_argument('--csv_path', type=str, required=True,
                       help='Path to CSV file')
    parser.add_argument('--h5_dir', type=str, required=True,
                       help='Path to H5 features directory')
    parser.add_argument('--label_col', type=str, default='disc_label',
                       help='Label column name')
    
    # 模型参数
    parser.add_argument('--in_dim', type=int, default=512,
                       help='Input feature dimension')
    parser.add_argument('--n_classes', type=int, default=4,
                       help='Number of survival time intervals')
    parser.add_argument('--dropout', type=float, default=0.25,
                       help='Dropout rate')
    parser.add_argument('--act', type=str, default='gelu',
                       choices=['relu', 'gelu'],
                       help='Activation function')
    parser.add_argument('--mamba_layer', type=int, default=2,
                       help='Number of Mamba layers')
    
    # 训练参数
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw'],
                       help='Optimizer')
    parser.add_argument('--loss', type=str, default='nll',
                       choices=['cox', 'nll'],
                       help='Loss function')
    parser.add_argument('--alpha_surv', type=float, default=0.0,
                       help='Alpha for NLL survival loss')
    parser.add_argument('--gc', type=int, default=1,
                       help='Gradient accumulation steps')
    
    # 早停参数
    parser.add_argument('--warmup', type=int, default=5,
                       help='Warmup epochs for early stopping')
    parser.add_argument('--patience', type=int, default=15,
                       help='Patience for early stopping')
    parser.add_argument('--stop_epoch', type=int, default=20,
                       help='Minimum epoch for early stopping')
    
    # K-Fold参数
    parser.add_argument('--k_fold', type=int, default=5,
                       help='Number of folds')
    parser.add_argument('--fold', type=int, default=None,
                       help='Specific fold to train (None for all folds)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Test set ratio')
    
    # 其他参数
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Results directory')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--weight_decay', type=float, default=1e-3,  # 从 1e-5 改为 1e-3
                    help='Weight decay for optimizer')
    parser.add_argument('--dropout', type=float, default=0.5,  # 从 0.25 改为 0.5
                    help='Dropout rate')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建结果目录
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(args.results_dir, 'config.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')
    
    # 训练
    if args.fold is not None:
        # 训练单个fold
        results = train_survival(args)
    else:
        # K-Fold交叉验证
        summary = train_k_fold(args)

