"""
生存分析训练框架 - DDP 多GPU版本
支持 Mamba2MIL 的分布式训练
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


# ===================== DDP 工具函数 =====================
def setup_ddp():
    """初始化 DDP 环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # 初始化进程组
        dist.init_process_group(backend='nccl')
        
        # 设置当前进程的GPU
        torch.cuda.set_device(local_rank)
        
        return rank, local_rank, world_size
    else:
        # 单GPU模式
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
            if self.verbose:
                print_rank0(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_cindex, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_cindex, model, ckpt_name):
        """保存模型检查点 - 只在主进程保存"""
        if not is_main_process():
            return
            
        if self.verbose:
            print_rank0(f'C-Index increased ({self.best_cindex:.4f} --> {val_cindex:.4f}). Saving model...')
        
        # 处理 DDP 模型
        if isinstance(model, DDP):
            torch.save(model.module.state_dict(), ckpt_name)
        else:
            torch.save(model.state_dict(), ckpt_name)
        
        self.best_cindex = val_cindex


# ===================== 数据加载器 =====================
def get_split_loader(split_dataset, batch_size=1, num_workers=4, training=False, 
                     use_ddp=False, world_size=1, rank=0):
    """
    创建数据加载器 - DDP 版本
    
    Args:
        split_dataset: 数据集
        batch_size: 批大小
        num_workers: 工作进程数
        training: 是否是训练模式
        use_ddp: 是否使用 DDP
        world_size: 总进程数
        rank: 当前进程rank
    """
    from dataset.dataset_h5 import custom_collate_fn
    
    if training:
        # 训练模式
        if use_ddp:
            # DDP: 使用 DistributedSampler
            sampler = DistributedSampler(
                split_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=False
            )
            loader = DataLoader(
                split_dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                collate_fn=custom_collate_fn,
                pin_memory=True,
                drop_last=False
            )
        else:
            # 单GPU: 正常 shuffle
            loader = DataLoader(
                split_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=custom_collate_fn,
                pin_memory=True,
                drop_last=False
            )
    else:
        # 验证/测试模式: batch_size=1, 不需要分布式
        loader = DataLoader(
            split_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=None,
            pin_memory=True,
            drop_last=False
        )
    
    return loader


# ===================== 训练循环 =====================
def train_loop(epoch, model, loader, optimizer, loss_fn, device, gc=1, 
               use_ddp=False, rank=0):
    """训练一个epoch - DDP 版本"""
    model.train()
    train_loss = 0.
    
    # 如果使用 DDP，设置 epoch 用于 shuffle
    if use_ddp and hasattr(loader.sampler, 'set_epoch'):
        loader.sampler.set_epoch(epoch)
    
    # 如果是组合损失,记录各分量
    is_combined = hasattr(loss_fn, 'get_loss_components')
    if is_combined:
        main_losses = []
        ranking_losses = []
    
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    
    # 只在主进程显示进度条
    if is_main_process():
        pbar = tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch} [Train]')
    else:
        pbar = enumerate(loader)
    
    for batch_idx, batch_data in pbar:
        if batch_data is None:
            continue
        
        (patient_list, gender, age, label, sur_time, censor, 
         features, coords, num_patches, mask) = batch_data
        
        features = features.to(device)
        mask = mask.to(device)
        label = label.to(device)
        censor = censor.to(device)
        
        batch_size = features.size(0)
        
        # 前向传播
        hazards, S, Y_hat, A, h = model(features, mask=mask)
        
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
        
        # 记录损失分量
        if is_combined:
            loss_components = loss_fn.get_loss_components(hazards, S, label, censor)
            main_losses.append(loss_components['main_loss'])
            ranking_losses.append(loss_components['ranking_loss'])
        
        # 计算风险分数
        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores.extend(risk)
        all_censorships.extend(censor.cpu().numpy())
        all_event_times.extend(sur_time.numpy())
        
        train_loss += loss_value
        
        # 反向传播
        loss = loss / gc
        loss.backward()
        
        if (batch_idx + 1) % gc == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # 更新进度条 (只在主进程)
        if is_main_process():
            if is_combined:
                pbar.set_postfix({
                    'loss': f'{loss_value:.4f}',
                    'main': f'{loss_components["main_loss"]:.4f}',
                    'rank': f'{loss_components["ranking_loss"]:.4f}'
                })
            else:
                pbar.set_postfix({'loss': f'{loss_value:.4f}'})
    
    # 最后一步
    if len(loader) % gc != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    # 🔥 DDP: 同步所有进程的指标
    if use_ddp:
        # 收集所有 GPU 的结果
        train_loss_tensor = torch.tensor([train_loss], device=device)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        train_loss = train_loss_tensor.item() / dist.get_world_size()
        
        # 收集所有 GPU 的预测结果
        all_risk_scores = torch.tensor(all_risk_scores, device=device)
        all_censorships = torch.tensor(all_censorships, device=device)
        all_event_times = torch.tensor(all_event_times, device=device)
        
        # gather 到主进程
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
    
    # 计算 C-Index (只在主进程)
    if is_main_process():
        c_index = concordance_index_censored(
            (1 - all_censorships).astype(bool),
            all_event_times,
            all_risk_scores,
            tied_tol=1e-08
        )[0]
        
        # 打印详细信息
        if is_combined:
            print_rank0(f'Epoch {epoch}: train_loss={train_loss:.4f} '
                  f'(main={np.mean(main_losses):.4f}, rank={np.mean(ranking_losses):.4f}), '
                  f'train_c_index={c_index:.4f}')
        else:
            print_rank0(f'Epoch {epoch}: train_loss={train_loss:.4f}, train_c_index={c_index:.4f}')
    else:
        c_index = 0.0
    
    # 广播 c_index 到所有进程
    if use_ddp:
        c_index_tensor = torch.tensor([c_index], device=device)
        dist.broadcast(c_index_tensor, src=0)
        c_index = c_index_tensor.item()
    
    return train_loss, c_index


# ===================== 验证循环 =====================
def validate(epoch, model, loader, loss_fn, device):
    """验证一个epoch - 只在主进程运行"""
    if not is_main_process():
        return 0.0, 0.0
    
    model.eval()
    val_loss = 0.
    
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch} [Val]')
    
    with torch.no_grad():
        for batch_idx, batch in pbar:
            features = batch['features'].to(device)
            label = batch['label'].to(device)
            event_time = batch['survival_time']
            c = batch['censorship'].to(device)
            
            # 前向传播
            hazards, S, Y_hat, _, _ = model(features)
            
            # 计算损失
            loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)
            loss_value = loss.item()
            
            # 计算风险分数
            risk = -torch.sum(S, dim=1).cpu().numpy()[0]
            all_risk_scores.append(risk)
            all_censorships.append(c.item())
            all_event_times.append(event_time.item())
            
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


# ===================== 测试函数 =====================
def test(model, loader, device):
    """在测试集上评估模型 - 只在主进程运行"""
    if not is_main_process():
        return {}, 0.0
    
    model.eval()
    
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    patient_results = {}
    
    pbar = tqdm(enumerate(loader), total=len(loader), desc='Testing')
    
    with torch.no_grad():
        for batch_idx, batch in pbar:
            case_id = batch['case_id'][0]
            features = batch['features'].to(device)
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
    
    c_index = concordance_index_censored(
        (1 - np.array(all_censorships)).astype(bool),
        np.array(all_event_times),
        np.array(all_risk_scores),
        tied_tol=1e-08
    )[0]
    
    print_rank0(f'Test C-Index: {c_index:.4f}')
    
    return patient_results, c_index


# ===================== 主训练函数 =====================
def train_survival(args):
    """主训练函数 - DDP 版本"""
    
    # 🔥 Step 1: 初始化 DDP
    rank, local_rank, world_size = setup_ddp()
    use_ddp = world_size > 1
    
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
    
    # 🔥 设置 Triton 缓存目录（避免多进程冲突）
    os.environ['TRITON_CACHE_DIR'] = f'/tmp/triton_cache_rank_{rank}'
    
    print_rank0('\n' + '='*60)
    print_rank0(f'Training Fold {args.fold}')
    if use_ddp:
        print_rank0(f'Using DDP with {world_size} GPUs (Rank {rank}/{world_size})')
    print_rank0('='*60)
    
    # 创建结果目录 (只在主进程)
    fold_dir = os.path.join(args.results_dir, f'fold_{args.fold}')
    if is_main_process():
        os.makedirs(fold_dir, exist_ok=True)
    
    # 🔥 DDP: 同步所有进程
    if use_ddp:
        dist.barrier()
    
    # ========== 1. 加载数据集 ==========
    print_rank0('\n[1/7] Loading dataset...')
    from dataset.dataset_h5 import PrognosisDataset
    
    dataset = PrognosisDataset(
        csv_path=args.csv_path,
        h5_dir=args.h5_dir,
        label_col=args.label_col,
        use_cache=True,
        print_info=is_main_process()
    )
    
    # 创建K-fold分割
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
    
    # 加载外部测试集
    external_test_dataset = None
    if hasattr(args, 'external_csv_path') and args.external_csv_path:
        print_rank0('\n[1.5/7] Loading External Test Set...')
        external_test_dataset = dataset.load_external_test(
            csv_path=args.external_csv_path,
            h5_dir=args.external_h5_dir
        )
    
    print_rank0(f'\nDataset sizes:')
    print_rank0(f'  Train: {len(train_dataset)} patients')
    print_rank0(f'  Val: {len(val_dataset)} patients')
    print_rank0(f'  Test: {len(test_dataset)} patients')
    if external_test_dataset:
        print_rank0(f'  External Test: {len(external_test_dataset)} patients')
    
    # ========== 2. 创建数据加载器 (DDP 版本) ==========
    print_rank0('\n[2/7] Creating data loaders...')
    
    train_loader = get_split_loader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        training=True,
        use_ddp=use_ddp,
        world_size=world_size,
        rank=rank
    )
    
    # 验证/测试只在主进程运行
    if is_main_process():
        val_loader = get_split_loader(
            val_dataset,
            batch_size=1,
            num_workers=args.num_workers,
            training=False
        )
        
        test_loader = get_split_loader(
            test_dataset,
            batch_size=1,
            num_workers=args.num_workers,
            training=False
        )
        
        if external_test_dataset is not None:
            external_test_loader = get_split_loader(
                external_test_dataset,
                batch_size=1,
                num_workers=args.num_workers,
                training=False
            )
        else:
            external_test_loader = None
    else:
        val_loader = None
        test_loader = None
        external_test_loader = None
    
    print_rank0(f'Train: {len(train_dataset)} samples, {len(train_loader)} batches (batch_size={args.batch_size})')
    if is_main_process():
        print_rank0(f'Val: {len(val_dataset)} samples, {len(val_loader)} batches')
        print_rank0(f'Test: {len(test_dataset)} samples, {len(test_loader)} batches')
        if external_test_loader:
            print_rank0(f'External: {len(external_test_dataset)} samples, {len(external_test_loader)} batches')
    
    # ========== 3. 初始化模型 (DDP 版本) ==========
    print_rank0('\n[3/7] Initializing model...')
    from models.Mamba2MIL import Mamba2MIL
    
    model = Mamba2MIL(
        in_dim=args.in_dim,
        n_classes=args.n_classes,
        dropout=args.dropout,
        act=args.act,
        survival=True,
        layer=args.mamba_layer,
        use_clinical=False
    )
    
    model = model.to(device)
    
    # 🔥 Step 2: 初始化 Triton kernels (避免 DDP 冲突)
    print_rank0('Initializing Triton kernels with dummy forward pass...')
    with torch.no_grad():
        dummy_input = torch.randn(1, 100, args.in_dim).to(device)
        try:
            _ = model(dummy_input)
            print_rank0('✓ Triton kernels initialized successfully')
        except Exception as e:
            print_rank0(f'⚠️  Triton initialization warning: {e}')
            print_rank0('   Continuing anyway...')
    
    # 🔥 DDP: 同步所有进程
    if use_ddp:
        dist.barrier()
    
    # 🔥 Step 3: 使用 DDP 包装模型
    if use_ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False  # Mamba2 不需要
        )
        print_rank0(f'✓ Using DDP with {world_size} GPUs')
    
    print_rank0(f'Model: Mamba2MIL')
    
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
    
    # 初始化损失函数
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
        print_rank0(f'Loss: Combined (NLL + {args.ranking_weight}*Ranking, alpha={args.alpha_surv})')
    
    print_rank0(f'Optimizer: {args.optimizer}, LR: {args.lr}, Weight Decay: {args.weight_decay}')
    
    # ========== 5. 训练循环 ==========
    print_rank0('\n[5/7] Training...')
    
    early_stopping = EarlyStopping(
        warmup=args.warmup,
        patience=args.patience,
        stop_epoch=args.stop_epoch,
        verbose=True
    )
    
    history = {
        'train_loss': [],
        'train_cindex': [],
        'val_loss': [],
        'val_cindex': []
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
            rank=rank
        )
        
        # 🔥 DDP: 同步所有进程
        if use_ddp:
            dist.barrier()
        
        # 验证 (只在主进程)
        val_loss, val_cindex = validate(
            epoch=epoch,
            model=model.module if use_ddp else model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device
        )
        
        # 🔥 DDP: 广播验证结果到所有进程
        if use_ddp:
            val_cindex_tensor = torch.tensor([val_cindex], device=device)
            dist.broadcast(val_cindex_tensor, src=0)
            val_cindex = val_cindex_tensor.item()
        
        # 记录历史 (只在主进程)
        if is_main_process():
            history['train_loss'].append(train_loss)
            history['train_cindex'].append(train_cindex)
            history['val_loss'].append(val_loss)
            history['val_cindex'].append(val_cindex)
        
        # 保存最佳模型 (只在主进程)
        if is_main_process():
            if val_cindex > best_val_cindex:
                best_val_cindex = val_cindex
                if isinstance(model, DDP):
                    torch.save(model.module.state_dict(), os.path.join(fold_dir, 'best_model.pt'))
                else:
                    torch.save(model.state_dict(), os.path.join(fold_dir, 'best_model.pt'))
                print_rank0(f'✓ Best model saved (val_cindex={val_cindex:.4f})')
        
        # 早停检查
        ckpt_path = os.path.join(fold_dir, 'checkpoint.pt')
        early_stopping(epoch, val_cindex, model, ckpt_name=ckpt_path)
        
        # 🔥 DDP: 广播早停信号
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
    
    # 保存训练历史 (只在主进程)
    if is_main_process():
        with open(os.path.join(fold_dir, 'history.pkl'), 'wb') as f:
            pickle.dump(history, f)
    
    # ========== 6. 测试 (只在主进程) ==========
    print_rank0('\n[6/7] Testing...')
    
    if is_main_process():
        # 🔥 Step 1: 获取单 GPU 模型
        if use_ddp:
            model_single = model.module
        else:
            model_single = model
        
        # 🔥 Step 2: 加载最佳模型
        best_model_path = os.path.join(fold_dir, 'best_model.pt')
        model_single.load_state_dict(torch.load(best_model_path))
        model_single.eval()
        
        print_rank0(f'Loaded best model from: {best_model_path}')
        
        # 🔥 Step 3: 在验证集上评估 (单 GPU)
        print_rank0('\nEvaluating on validation set...')
        val_loader_test = get_split_loader(
            val_dataset,
            batch_size=1,
            num_workers=args.num_workers,
            training=False,
            use_ddp=False,
            world_size=1,
            rank=0
        )
        
        val_results, val_cindex = test(model_single, val_loader_test, device)
        print_rank0(f'Validation C-Index: {val_cindex:.4f}')
        
        # 🔥 Step 4: 在内部测试集上评估 (单 GPU)
        print_rank0('\nEvaluating on internal test set...')
        test_loader_test = get_split_loader(
            test_dataset,
            batch_size=1,
            num_workers=args.num_workers,
            training=False,
            use_ddp=False,
            world_size=1,
            rank=0
        )
        
        test_results, test_cindex = test(model_single, test_loader_test, device)
        print_rank0(f'Internal Test C-Index: {test_cindex:.4f}')
        
        # 🔥 Step 5: 在外部测试集上评估 (单 GPU)
        external_test_results = None
        external_test_cindex = None
        
        if hasattr(args, 'external_csv_path') and args.external_csv_path:
            print_rank0('\n[7/7] Evaluating on External Test Set...')
            
            # 重新加载外部测试集
            from dataset.dataset_h5 import PrognosisDataset
            
            external_test_dataset_reload = PrognosisDataset(
                csv_path=args.external_csv_path,
                h5_dir=args.external_h5_dir,
                label_col=args.label_col,
                use_cache=True,
                print_info=False
            )
            
            external_test_loader_test = get_split_loader(
                external_test_dataset_reload,
                batch_size=1,
                num_workers=args.num_workers,
                training=False,
                use_ddp=False,
                world_size=1,
                rank=0
            )
            
            external_test_results, external_test_cindex = test(
                model_single,
                external_test_loader_test,
                device
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
    
    # 🔥 同步所有进程
    if use_ddp:
        dist.barrier()
    
    # 🔥 清理 DDP
    cleanup_ddp()
    
    return results


# ===================== K-Fold交叉验证 =====================
def train_k_fold(args):
    """K-Fold交叉验证 - DDP 版本"""
    
    # 只在主进程运行 K-Fold
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
