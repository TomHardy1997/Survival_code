"""
生存分析训练框架 - 抗过拟合增强版 + DDP优化 + 临床特征支持
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from sksurv.metrics import concordance_index_censored
import pandas as pd
from tqdm import tqdm
import pickle
from argparse import Namespace
import warnings


# ===================== DDP 工具函数 =====================
def setup_ddp():
    """初始化 DDP 环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return rank, local_rank, world_size
    else:
        return 0, 0, 1


def cleanup_ddp():
    """清理 DDP"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """判断是否是主进程"""
    return not dist.is_initialized() or dist.get_rank() == 0


def print_rank0(*args, **kwargs):
    """只在主进程打印"""
    if is_main_process():
        print(*args, **kwargs)


# ===================== 早停机制 (改进版) =====================
class EarlyStopping:
    """
    基于C-Index的早停 - 改进版
    
    新增功能:
    1. 支持学习率衰减
    2. 保存多个检查点
    3. 更灵活的patience策略
    """
    def __init__(self, warmup=5, patience=15, stop_epoch=20, verbose=False,
                 delta=0.0001, save_all=False):
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.delta = delta
        self.save_all = save_all
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_cindex = 0
        self.checkpoint_history = []

    def __call__(self, epoch, val_cindex, model, ckpt_name='checkpoint.pt'):
        score = val_cindex

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_cindex, model, ckpt_name, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print_rank0(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_cindex, model, ckpt_name, epoch)
            self.counter = 0

    def save_checkpoint(self, val_cindex, model, ckpt_name, epoch):
        """保存模型检查点"""
        if not is_main_process():
            return
            
        if self.verbose:
            print_rank0(f'C-Index increased ({self.best_cindex:.4f} --> {val_cindex:.4f}). Saving model...')
        
        # 处理 DDP 模型
        state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
        
        # 保存主检查点
        torch.save(state_dict, ckpt_name)
        
        # 保存带epoch的检查点
        if self.save_all:
            epoch_ckpt = ckpt_name.replace('.pt', f'_epoch{epoch}.pt')
            torch.save(state_dict, epoch_ckpt)
            self.checkpoint_history.append({
                'epoch': epoch,
                'cindex': val_cindex,
                'path': epoch_ckpt
            })
        
        self.best_cindex = val_cindex


# ===================== 梯度裁剪工具 =====================
def clip_gradients(model, max_norm=1.0):
    """梯度裁剪 - 防止梯度爆炸"""
    if isinstance(model, DDP):
        torch.nn.utils.clip_grad_norm_(model.module.parameters(), max_norm)
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


# ===================== 学习率调度器 =====================
def get_scheduler(optimizer, args):
    """
    创建学习率调度器
    
    支持:
    - cosine: 余弦退火
    - step: 阶梯衰减
    - plateau: 基于验证指标的自适应衰减
    """
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.max_epochs,
            eta_min=args.lr * 0.01
        )
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma
        )
    elif args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=args.lr * 0.001
        )
    else:
        scheduler = None
    
    return scheduler


# ===================== 数据增强 =====================
def augment_features(features, mask, training=True, drop_rate=0.1):
    """
    特征级数据增强
    
    Args:
        features: [B, N, D]
        mask: [B, N]
        training: 是否训练模式
        drop_rate: patch丢弃率
    """
    if not training or drop_rate == 0:
        return features, mask
    
    B, N, D = features.shape
    
    # 随机丢弃部分patch
    keep_mask = torch.rand(B, N, device=features.device) > drop_rate
    keep_mask = keep_mask & (mask > 0)
    
    # 确保每个样本至少保留一个patch
    for i in range(B):
        if keep_mask[i].sum() == 0:
            valid_indices = torch.where(mask[i] > 0)[0]
            if len(valid_indices) > 0:
                keep_mask[i, valid_indices[0]] = True
    
    return features, keep_mask.float()


# ===================== 训练循环 (改进版) =====================
def train_loop(epoch, model, loader, optimizer, loss_fn, device, gc=1, 
               use_ddp=False, rank=0, scheduler=None, max_grad_norm=1.0,
               feature_drop_rate=0.0, use_clinical=False):
    """
    训练一个epoch - 改进版
    
    🔥 新增参数:
        use_clinical: 是否使用临床特征
    """
    model.train()
    train_loss = 0.
    
    if use_ddp and hasattr(loader.sampler, 'set_epoch'):
        loader.sampler.set_epoch(epoch)
    
    is_combined = hasattr(loss_fn, 'get_loss_components')
    if is_combined:
        main_losses = []
        ranking_losses = []
    
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    
    if is_main_process():
        pbar = tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch} [Train]')
    else:
        pbar = enumerate(loader)
    
    for batch_idx, batch_data in pbar:
        if batch_data is None:
            continue
        
        # 🔥 提取所有数据
        features = batch_data['features'].to(device)
        mask = batch_data['mask'].to(device)
        label = batch_data['label'].to(device)
        censor = batch_data['censorship'].to(device)
        sur_time = batch_data['survival_time']  # CPU
        
        # 🔥 提取临床特征 (如果使用)
        gender = None
        age = None
        if use_clinical:
            if 'gender' in batch_data:
                gender = batch_data['gender'].to(device)
            if 'age' in batch_data:
                age = batch_data['age'].to(device)
        
        # 特征级数据增强
        if feature_drop_rate > 0:
            features, mask = augment_features(features, mask, training=True, 
                                             drop_rate=feature_drop_rate)
        
        batch_size = features.size(0)
        
        # 🔥 前向传播 - 传入临床特征
        hazards, S, Y_hat, A, h = model(
            x=features, 
            mask=mask,
            gender=gender,  # ✅ 传入性别
            age=age         # ✅ 传入年龄
        )
        
        # 计算损失
        loss = 0
        for i in range(batch_size):
            loss += loss_fn(
                hazards=hazards[i:i+1],
                S=S[i:i+1],
                Y=label[i:i+1],
                c=censor[i:i+1]
            )
        loss = loss / batch_size
        loss_value = loss.item()
        
        if is_combined:
            loss_components = loss_fn.get_loss_components(hazards, S, label, censor)
            main_losses.append(loss_components['main_loss'])
            ranking_losses.append(loss_components['ranking_loss'])
        
        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores.extend(risk)
        all_censorships.extend(censor.cpu().numpy())
        all_event_times.extend(sur_time.numpy())
        
        train_loss += loss_value
        
        # 反向传播
        loss = loss / gc
        loss.backward()
        
        if (batch_idx + 1) % gc == 0:
            clip_gradients(model, max_norm=max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
        
        if is_main_process():
            if is_combined:
                pbar.set_postfix({
                    'loss': f'{loss_value:.4f}',
                    'main': f'{loss_components["main_loss"]:.4f}',
                    'rank': f'{loss_components["ranking_loss"]:.4f}'
                })
            else:
                pbar.set_postfix({'loss': f'{loss_value:.4f}'})
    
    if len(loader) % gc != 0:
        clip_gradients(model, max_norm=max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
    
    # DDP: 同步指标
    if use_ddp:
        train_loss_tensor = torch.tensor([train_loss], device=device)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        train_loss = train_loss_tensor.item() / dist.get_world_size()
        
        all_risk_scores = torch.tensor(all_risk_scores, device=device)
        all_censorships = torch.tensor(all_censorships, device=device)
        all_event_times = torch.tensor(all_event_times, device=device)
        
        if is_main_process():
            gathered_risks = [torch.zeros_like(all_risk_scores) for _ in range(dist.get_world_size())]
            gathered_censors = [torch.zeros_like(all_censorships) for _ in range(dist.get_world_size())]
            gathered_times = [torch.zeros_like(all_event_times) for _ in range(dist.get_world_size())]
        else:
            gathered_risks = None
            gathered_censors = None
            gathered_times = None
        
        dist.gather(all_risk_scores, gathered_risks, dst=0)
        dist.gather(all_censorships, gathered_censors, dst=0)
        dist.gather(all_event_times, gathered_times, dst=0)
        
        if is_main_process():
            all_risk_scores = torch.cat(gathered_risks).cpu().numpy()
            all_censorships = torch.cat(gathered_censors).cpu().numpy()
            all_event_times = torch.cat(gathered_times).cpu().numpy()
    else:
        train_loss /= len(loader)
        all_risk_scores = np.array(all_risk_scores)
        all_censorships = np.array(all_censorships)
        all_event_times = np.array(all_event_times)
    
    # 计算 C-Index
    if is_main_process():
        c_index = concordance_index_censored(
            (1 - all_censorships).astype(bool),
            all_event_times,
            all_risk_scores,
            tied_tol=1e-08
        )[0]
        
        current_lr = optimizer.param_groups[0]['lr']
        
        if is_combined:
            print_rank0(f'Epoch {epoch}: train_loss={train_loss:.4f} '
                  f'(main={np.mean(main_losses):.4f}, rank={np.mean(ranking_losses):.4f}), '
                  f'train_c_index={c_index:.4f}, lr={current_lr:.2e}')
        else:
            print_rank0(f'Epoch {epoch}: train_loss={train_loss:.4f}, '
                       f'train_c_index={c_index:.4f}, lr={current_lr:.2e}')
    else:
        c_index = 0.0
    
    if use_ddp:
        c_index_tensor = torch.tensor([c_index], device=device)
        dist.broadcast(c_index_tensor, src=0)
        c_index = c_index_tensor.item()
    
    return train_loss, c_index


# ===================== 验证循环 (改进版) =====================
def validate(epoch, model, loader, loss_fn, device, use_clinical=False):
    """
    验证一个epoch
    
    🔥 新增参数:
        use_clinical: 是否使用临床特征
    """
    if not is_main_process():
        return 0.0, 0.0
    
    model.eval()
    val_loss = 0.
    
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch} [Val]')
    
    with torch.no_grad():
        for batch_idx, batch_data in pbar:
            if batch_data is None:
                continue
            
            # 🔥 提取所有数据
            features = batch_data['features'].to(device)
            mask = batch_data['mask'].to(device)
            label = batch_data['label'].to(device)
            censor = batch_data['censorship'].to(device)
            event_time = batch_data['survival_time']  # CPU
            
            # 🔥 提取临床特征 (如果使用)
            gender = None
            age = None
            if use_clinical:
                if 'gender' in batch_data:
                    gender = batch_data['gender'].to(device)
                if 'age' in batch_data:
                    age = batch_data['age'].to(device)
            
            # 🔥 前向传播 - 传入临床特征
            hazards, S, Y_hat, _, _ = model(
                x=features,
                mask=mask,
                gender=gender,  # ✅ 传入性别
                age=age         # ✅ 传入年龄
            )
            
            # 计算损失
            loss = loss_fn(hazards=hazards, S=S, Y=label, c=censor)
            loss_value = loss.item()
            
            risk = -torch.sum(S, dim=1).cpu().numpy()
            all_risk_scores.extend(risk)
            all_censorships.extend(censor.cpu().numpy())
            all_event_times.extend(event_time.numpy())
            
            val_loss += loss_value
            pbar.set_postfix({'loss': f'{loss_value:.4f}'})
    
    val_loss /= len(loader)
    c_index = concordance_index_censored(
        (1 - np.array(all_censorships)).astype(bool),
        np.array(all_event_times),
        np.array(all_risk_scores),
        tied_tol=1e-08
    )[0]
    
    print_rank0(f'Epoch {epoch}: val_loss={val_loss:.4f}, val_c_index={c_index:.4f}')
    
    return val_loss, c_index


# ===================== 测试函数 (改进版) =====================
def test(model, loader, device, use_clinical=False):
    """
    在测试集上评估模型
    
    🔥 新增参数:
        use_clinical: 是否使用临床特征
    """
    if not is_main_process():
        return {}, 0.0
    
    model.eval()
    
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    patient_results = {}
    
    pbar = tqdm(enumerate(loader), total=len(loader), desc='Testing')
    
    with torch.no_grad():
        for batch_idx, batch_data in pbar:
            if batch_data is None:
                continue
            
            # 🔥 提取所有数据
            case_id = batch_data['case_id'][0]
            features = batch_data['features'].to(device)
            mask = batch_data['mask'].to(device)
            label = batch_data['label']
            event_time = batch_data['survival_time']
            c = batch_data['censorship']
            
            # 🔥 提取临床特征 (如果使用)
            gender = None
            age = None
            if use_clinical:
                if 'gender' in batch_data:
                    gender = batch_data['gender'].to(device)
                if 'age' in batch_data:
                    age = batch_data['age'].to(device)
            
            # 🔥 前向传播 - 传入临床特征
            hazards, S, Y_hat, _, _ = model(
                x=features,
                mask=mask,
                gender=gender,  # ✅ 传入性别
                age=age         # ✅ 传入年龄
            )
            
            risk = -torch.sum(S, dim=1).cpu().numpy()[0]
            
            all_risk_scores.append(risk)
            all_censorships.append(c.item())
            all_event_times.append(event_time.item())
            
            patient_results[case_id] = {
                'case_id': case_id,
                'risk': risk,
                'disc_label': label.item(),
                'survival': event_time.item(),
                'censorship': c.item(),
                'hazards': hazards.cpu().numpy(),
                'S': S.cpu().numpy()
            }
    
    c_index = concordance_index_censored(
        (1 - np.array(all_censorships)).astype(bool),
        np.array(all_event_times),
        np.array(all_risk_scores),
        tied_tol=1e-08
    )[0]
    
    print_rank0(f'Test C-Index: {c_index:.4f}')
    
    return patient_results, c_index


# ===================== 主训练函数 (改进版) =====================
def train_survival(args):
    """主训练函数 - 改进版"""
    
    # 初始化 DDP
    rank, local_rank, world_size = setup_ddp()
    use_ddp = world_size > 1
    
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
    
    os.environ['TRITON_CACHE_DIR'] = f'/tmp/triton_cache_rank_{rank}'
    
    print_rank0('\n' + '='*60)
    print_rank0(f'Training Fold {args.fold}')
    if use_ddp:
        print_rank0(f'Using DDP with {world_size} GPUs (Rank {rank}/{world_size})')
    print_rank0('='*60)
    
    fold_dir = os.path.join(args.results_dir, f'fold_{args.fold}')
    if is_main_process():
        os.makedirs(fold_dir, exist_ok=True)
    
    if use_ddp:
        dist.barrier()
    
    # ========== 1. 加载数据集 ==========
    print_rank0('\n[1/7] Loading dataset...')
    from dataset.dataset_xiugai import PrognosisDataset, custom_collate_fn
    
    dataset = PrognosisDataset(
        csv_path=args.csv_path,
        h5_base_dir=args.h5_base_dir,
        feature_models=args.feature_models,
        label_col=args.label_col,
        use_cache=True,
        normalize_age=args.normalize_age if hasattr(args, 'normalize_age') else True,  # 🔥 新增
        print_info=is_main_process()
    )
    
    if not hasattr(dataset, 'splits'):
        dataset.create_splits(
            n_splits=args.k_fold,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            stratify=True
        )
    
    dataset.set_split(fold=args.fold)
    
    train_dataset = dataset.get_split_dataset('train')
    val_dataset = dataset.get_split_dataset('val')
    test_dataset = dataset.get_split_dataset('test')
    
    external_test_dataset = None
    if hasattr(args, 'external_csv_path') and args.external_csv_path:
        print_rank0('\n[1.5/7] Loading External Test Set...')
        external_test_dataset = dataset.load_external_test(
            csv_path=args.external_csv_path,
            h5_base_dir=args.external_h5_base_dir,
            feature_models=args.feature_models
        )
    
    print_rank0(f'\nDataset sizes:')
    print_rank0(f'  Train: {len(train_dataset)} patients')
    print_rank0(f'  Val: {len(val_dataset)} patients')
    print_rank0(f'  Test: {len(test_dataset)} patients')
    if external_test_dataset:
        print_rank0(f'  External Test: {len(external_test_dataset)} patients')
    
    # ========== 2. 创建数据加载器 ==========
    print_rank0('\n[2/7] Creating data loaders...')
    
    if use_ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True,
            drop_last=False
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True,
            drop_last=False
        )
    
    if is_main_process():
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=custom_collate_fn,  # 🔥 修复: 使用custom_collate_fn
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=custom_collate_fn,  # 🔥 修复: 使用custom_collate_fn
            pin_memory=True
        )
        
        if external_test_dataset is not None:
            external_test_loader = DataLoader(
                external_test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=custom_collate_fn,  # 🔥 修复: 使用custom_collate_fn
                pin_memory=True
            )
        else:
            external_test_loader = None
    else:
        val_loader = None
        test_loader = None
        external_test_loader = None
    
    # ========== 3. 初始化模型 ==========
    print_rank0('\n[3/7] Initializing model...')
    
    from models.Mamba2MIL2 import Mamba2MIL

    # 🔥 检查是否使用临床特征
    use_clinical = getattr(args, 'use_clinical', False)
    
    model = Mamba2MIL(
        in_dim=args.in_dim,
        n_classes=args.n_classes,
        dropout=args.dropout,
        act=args.act,
        survival=True,
        layer=args.mamba_layer,
        use_clinical=use_clinical  # 🔥 传入参数
    )
    
    print_rank0(f'Model configuration:')
    print_rank0(f'  Use clinical features: {use_clinical}')
    
    model = model.to(device)
    
    # 初始化 Triton kernels
    print_rank0('Initializing Triton kernels...')
    with torch.no_grad():
        dummy_input = torch.randn(1, 100, args.in_dim).to(device)
        try:
            if use_clinical:
                dummy_gender = torch.tensor([0], device=device)
                dummy_age = torch.tensor([0.0], device=device)
                _ = model(dummy_input, gender=dummy_gender, age=dummy_age)
            else:
                _ = model(dummy_input)
            print_rank0('✓ Triton kernels initialized')
        except Exception as e:
            print_rank0(f'⚠️  Warning: {e}')
    
    if use_ddp:
        dist.barrier()
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )
        print_rank0(f'✓ Using DDP with {world_size} GPUs')
    
    # 统计参数
    if isinstance(model, DDP):
        total_params = sum(p.numel() for p in model.module.parameters())
        trainable_params = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
    else:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print_rank0(f'  Parameters: {total_params:,}')
    print_rank0(f'  Trainable: {trainable_params:,}')
    
    # ========== 4. 初始化优化器和损失函数 ==========
    print_rank0('\n[4/7] Initializing optimizer and loss function...')
    
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
    
    # 学习率调度器
    scheduler = get_scheduler(optimizer, args)
    if scheduler:
        print_rank0(f'Using scheduler: {args.scheduler}')
    
    # 损失函数
    from utils.survival_loss_function import NLLSurvLoss, CoxSurvLoss, CombinedSurvLoss
    
    if args.loss == 'cox':
        loss_fn = CoxSurvLoss()
        print_rank0(f'Loss: Cox')
    elif args.loss == 'nll':
        loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
        print_rank0(f'Loss: NLL (alpha={args.alpha_surv})')
    elif args.loss == 'combined':
        loss_fn = CombinedSurvLoss(
            main_loss_type='nll',
            alpha=args.alpha_surv,
            ranking_weight=args.ranking_weight,
            ranking_margin=args.ranking_margin
        )
        print_rank0(f'Loss: Combined (NLL + {args.ranking_weight}*Ranking)')
    
    print_rank0(f'Optimizer: {args.optimizer}, LR: {args.lr}, Weight Decay: {args.weight_decay}')
    
    # ========== 5. 训练循环 ==========
    print_rank0('\n[5/7] Training...')
    
    early_stopping = EarlyStopping(
        warmup=args.warmup,
        patience=args.patience,
        stop_epoch=args.stop_epoch,
        verbose=True,
        delta=getattr(args, 'early_stop_delta', 0.0001),
        save_all=getattr(args, 'save_all_checkpoints', False)
    )
    
    history = {
        'train_loss': [],
        'train_cindex': [],
        'val_loss': [],
        'val_cindex': [],
        'lr': []
    }
    
    best_val_cindex = 0
    
    for epoch in range(args.max_epochs):
        print_rank0(f'\n{"="*60}')
        print_rank0(f'Epoch {epoch+1}/{args.max_epochs}')
        print_rank0(f'{"="*60}')
        
        # 训练
        train_loss, train_cindex = train_loop(
            epoch=epoch,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            gc=args.gc,
            use_ddp=use_ddp,
            rank=rank,
            scheduler=scheduler if args.scheduler != 'plateau' else None,
            max_grad_norm=getattr(args, 'max_grad_norm', 1.0),
            feature_drop_rate=getattr(args, 'feature_drop_rate', 0.0),
            use_clinical=use_clinical  # 🔥 传入参数
        )
        
        if use_ddp:
            dist.barrier()
        
        # 验证
        val_loss, val_cindex = validate(
            epoch=epoch,
            model=model.module if use_ddp else model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            use_clinical=use_clinical  # 🔥 传入参数
        )
        
        # 更新学习率调度器
        if scheduler:
            if args.scheduler == 'plateau':
                scheduler.step(val_cindex)
            else:
                scheduler.step()
        
        if use_ddp:
            val_cindex_tensor = torch.tensor([val_cindex], device=device)
            dist.broadcast(val_cindex_tensor, src=0)
            val_cindex = val_cindex_tensor.item()
        
        # 记录历史
        if is_main_process():
            history['train_loss'].append(train_loss)
            history['train_cindex'].append(train_cindex)
            history['val_loss'].append(val_loss)
            history['val_cindex'].append(val_cindex)
            history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # 保存最佳模型
        if is_main_process():
            if val_cindex > best_val_cindex:
                best_val_cindex = val_cindex
                state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                torch.save(state_dict, os.path.join(fold_dir, 'best_model.pt'))
                print_rank0(f'✓ Best model saved (val_cindex={val_cindex:.4f})')
        
        # 早停检查
        ckpt_path = os.path.join(fold_dir, 'checkpoint.pt')
        early_stopping(epoch, val_cindex, model, ckpt_name=ckpt_path)
        
        if use_ddp:
            early_stop_tensor = torch.tensor([1 if early_stopping.early_stop else 0], device=device)
            dist.broadcast(early_stop_tensor, src=0)
            if early_stop_tensor.item() == 1:
                print_rank0(f'\nEarly stopping at epoch {epoch+1}')
                break
        else:
            if early_stopping.early_stop:
                print_rank0(f'\nEarly stopping at epoch {epoch+1}')
                break
    
    # 保存训练历史
    if is_main_process():
        with open(os.path.join(fold_dir, 'history.pkl'), 'wb') as f:
            pickle.dump(history, f)
        
        # 保存学习率曲线
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_cindex'], label='Train')
        plt.plot(history['val_cindex'], label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('C-Index')
        plt.legend()
        plt.title('C-Index Curve')
        
        plt.subplot(1, 2, 2)
        plt.plot(history['lr'])
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.title('Learning Rate Schedule')
        
        plt.tight_layout()
        plt.savefig(os.path.join(fold_dir, 'training_curves.png'), dpi=150)
        plt.close()
    
    # ========== 6. 测试 ==========
    print_rank0('\n[6/7] Testing...')
    
    if is_main_process():
        model_single = model.module if use_ddp else model
        
        best_model_path = os.path.join(fold_dir, 'best_model.pt')
        model_single.load_state_dict(torch.load(best_model_path))
        model_single.eval()
        
        print_rank0(f'Loaded best model from: {best_model_path}')
        
        # 验证集
        print_rank0('\nEvaluating on validation set...')
        val_results, val_cindex = test(
            model_single, 
            val_loader, 
            device,
            use_clinical=use_clinical  # 🔥 传入参数
        )
        print_rank0(f'Validation C-Index: {val_cindex:.4f}')
        
        # 内部测试集
        print_rank0('\nEvaluating on internal test set...')
        test_results, test_cindex = test(
            model_single, 
            test_loader, 
            device,
            use_clinical=use_clinical  # 🔥 传入参数
        )
        print_rank0(f'Internal Test C-Index: {test_cindex:.4f}')
        
        # 外部测试集
        external_test_results = None
        external_test_cindex = None
        
        if external_test_loader is not None:
            print_rank0('\n[7/7] Evaluating on External Test Set...')
            external_test_results, external_test_cindex = test(
                model_single,
                external_test_loader,
                device,
                use_clinical=use_clinical  # 🔥 传入参数
            )
            print_rank0(f'External Test C-Index: {external_test_cindex:.4f}')
        
        # 保存结果
        results = {
            'fold': args.fold,
            'best_val_cindex': best_val_cindex,
            'val_cindex': val_cindex,
            'test_cindex': test_cindex,
            'val_results': val_results,
            'test_results': test_results,
            'history': history
        }
        
        if external_test_results is not None:
            results['external_cindex'] = external_test_cindex
            results['external_test_results'] = external_test_results
        
        with open(os.path.join(fold_dir, 'results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        
        # 保存CSV
        val_df = pd.DataFrame([v for v in val_results.values()])
        val_df.to_csv(os.path.join(fold_dir, 'val_results.csv'), index=False)
        
        test_df = pd.DataFrame([v for v in test_results.values()])
        test_df.to_csv(os.path.join(fold_dir, 'test_results.csv'), index=False)
        
        if external_test_results is not None:
            external_df = pd.DataFrame([v for v in external_test_results.values()])
            external_df.to_csv(os.path.join(fold_dir, 'external_test_results.csv'), index=False)
        
        print_rank0('\n' + '='*60)
        print_rank0('Training completed!')
        print_rank0('='*60)
        print_rank0(f'Best Validation C-Index: {best_val_cindex:.4f}')
        print_rank0(f'Final Validation C-Index: {val_cindex:.4f}')
        print_rank0(f'Internal Test C-Index: {test_cindex:.4f}')
        if external_test_cindex is not None:
            print_rank0(f'External Test C-Index: {external_test_cindex:.4f}')
        print_rank0(f'Results saved to: {fold_dir}')
    else:
        results = None
    
    if use_ddp:
        dist.barrier()
    
    cleanup_ddp()
    
    return results


# ===================== K-Fold交叉验证 (不变) =====================
def train_k_fold(args):
    """K-Fold交叉验证"""
    if not is_main_process():
        return None
    
    print_rank0('\n' + '='*60)
    print_rank0(f'K-Fold Cross Validation (K={args.k_fold})')
    print_rank0('='*60)
    
    all_results = []
    
    for fold in range(args.k_fold):
        args.fold = fold
        results = train_survival(args)
        if results is not None:
            all_results.append(results)
    
    # 汇总结果
    val_cindices = [r['val_cindex'] for r in all_results]
    test_cindices = [r['test_cindex'] for r in all_results]
    
    external_cindices = []
    has_external = False
    for r in all_results:
        if 'external_cindex' in r:
            external_cindices.append(r['external_cindex'])
            has_external = True
    
    print_rank0('\n' + '='*60)
    print_rank0('K-Fold Cross Validation Results')
    print_rank0('='*60)
    
    for fold in range(args.k_fold):
        print_rank0(f'Fold {fold}: Val={val_cindices[fold]:.4f}, Test={test_cindices[fold]:.4f}', end='')
        if has_external:
            print_rank0(f', External={external_cindices[fold]:.4f}')
        else:
            print_rank0()
    
    print_rank0(f'\nMean Val C-Index: {np.mean(val_cindices):.4f} ± {np.std(val_cindices):.4f}')
    print_rank0(f'Mean Test C-Index: {np.mean(test_cindices):.4f} ± {np.std(test_cindices):.4f}')
    if has_external:
        print_rank0(f'Mean External C-Index: {np.mean(external_cindices):.4f} ± {np.std(external_cindices):.4f}')
    
    # 保存汇总
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
    
    summary_data = {'fold': range(args.k_fold), 'val_cindex': val_cindices, 'test_cindex': test_cindices}
    if has_external:
        summary_data['external_cindex'] = external_cindices
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(args.results_dir, 'summary.csv'), index=False)
    
    print_rank0(f'\nSummary saved to: {args.results_dir}')
    
    return summary


# ===================== 测试代码 =====================
if __name__ == '__main__':
    print("="*60)
    print("Testing core_utils2.py")
    print("="*60)
    
    # 创建测试参数
    args = Namespace(
        # 数据相关
        csv_path='/home/stat-jijianxin/PFMs/Survival_code/csv_file/tcga_survival_matched.csv',
        h5_base_dir='/home/stat-jijianxin/PFMs/TRIDENT/tcga_filtered/20x_512px_0px_overlap',
        feature_models='uni_v1',
        label_col='disc_label',
        normalize_age=True,  # 🔥 新增
        
        # 外部测试集
        external_csv_path='/home/stat-jijianxin/PFMs/Survival_code/csv_file/hmu_survival_with_slides.csv',
        external_h5_base_dir='/home/stat-jijianxin/PFMs/TRIDENT/hmu_filtered/20x_512px_0px_overlap',
        
        # 模型相关
        in_dim=512,
        n_classes=4,
        dropout=0.25,
        act='relu',
        mamba_layer=2,
        use_clinical=True,  # 🔥 测试临床特征
        
        # 训练相关
        max_epochs=5,  # 测试用少量epoch
        batch_size=2,
        lr=2e-4,
        weight_decay=1e-5,
        optimizer='adamw',
        scheduler='cosine',
        gc=1,
        
        # 损失函数
        loss='nll',
        alpha_surv=0.15,
        
        # 早停
        warmup=2,
        patience=3,
        stop_epoch=3,
        
        # 数据分割
        k_fold=3,
        val_ratio=0.15,
        test_ratio=0.15,
        fold=0,
        
        # 其他
        results_dir='./test_results',
        num_workers=4,
        max_grad_norm=1.0,
        feature_drop_rate=0.1,
        early_stop_delta=0.0001,
        save_all_checkpoints=False
    )
    
    print("\n[Test 1] Testing single fold training...")
    try:
        results = train_survival(args)
        print("✓ Single fold training completed!")
        
        if results:
            print(f"\nResults:")
            print(f"  Val C-Index: {results['val_cindex']:.4f}")
            print(f"  Test C-Index: {results['test_cindex']:.4f}")
            if 'external_cindex' in results:
                print(f"  External C-Index: {results['external_cindex']:.4f}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)
