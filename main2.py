"""
生存分析训练主脚本 - 抗过拟合增强版
"""
import os
import sys
import argparse
import torch
import numpy as np
import json
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.core_utils2 import train_survival, train_k_fold


def main():
    parser = argparse.ArgumentParser(description='Survival Analysis Training - Enhanced Version')
    
    # ========== 数据参数 ==========
    data_group = parser.add_argument_group('Data Parameters')
    data_group.add_argument('--csv_path', type=str, required=True,
                           help='训练集CSV文件路径')
    data_group.add_argument('--h5_base_dir', type=str, required=True,
                           help='H5特征文件基础目录')
    data_group.add_argument('--feature_models', type=str, nargs='+', 
                           default=['uni_v1'],
                           help='特征提取模型列表 (支持多模型融合)')
    data_group.add_argument('--label_col', type=str, default='disc_label',
                           help='标签列名')
    
    # ========== 外部测试集参数 ==========
    external_group = parser.add_argument_group('External Test Set Parameters')
    external_group.add_argument('--external_csv_path', type=str, default=None,
                               help='外部测试集CSV文件路径 (可选)')
    external_group.add_argument('--external_h5_base_dir', type=str, default=None,
                               help='外部测试集H5特征文件基础目录 (可选)')
    
    # ========== 模型参数 ==========
    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument('--model_version', type=str, default='standard',
                            choices=['standard', 'lite'],
                            help='模型版本: standard(标准) 或 lite(轻量级)')
    model_group.add_argument('--in_dim', type=int, default=1024,
                            help='输入特征维度')
    model_group.add_argument('--n_classes', type=int, default=4,
                            help='生存时间区间数量')
    model_group.add_argument('--dropout', type=float, default=0.4,
                            help='Dropout比率 (推荐0.3-0.5)')
    model_group.add_argument('--drop_path_rate', type=float, default=0.1,
                            help='Stochastic Depth比率 (仅standard版本)')
    model_group.add_argument('--feature_dropout', type=float, default=0.1,
                            help='特征层Dropout比率 (仅standard版本)')
    model_group.add_argument('--act', type=str, default='gelu',
                            choices=['relu', 'gelu'],
                            help='激活函数')
    model_group.add_argument('--mamba_layer', type=int, default=2,
                            help='Mamba层数 (1-3, 过拟合时建议用1)')
    
    # ========== 训练参数 ==========
    train_group = parser.add_argument_group('Training Parameters')
    train_group.add_argument('--batch_size', type=int, default=4,
                            help='批大小 (DDP时会自动分配到各GPU)')
    train_group.add_argument('--max_epochs', type=int, default=100,
                            help='最大训练轮数')
    train_group.add_argument('--lr', type=float, default=5e-5,
                            help='学习率 (推荐1e-5到1e-4)')
    train_group.add_argument('--weight_decay', type=float, default=1e-3,
                            help='权重衰减/L2正则化 (推荐1e-4到1e-2)')
    train_group.add_argument('--optimizer', type=str, default='adamw',
                            choices=['adam', 'adamw'],
                            help='优化器 (推荐adamw)')
    train_group.add_argument('--loss', type=str, default='combined',
                            choices=['cox', 'nll', 'combined'],
                            help='损失函数类型')
    train_group.add_argument('--alpha_surv', type=float, default=0.0,
                            help='NLL损失的alpha参数')
    train_group.add_argument('--gc', type=int, default=1,
                            help='梯度累积步数')
    
    # ========== 🔥 新增: 正则化参数 ==========
    reg_group = parser.add_argument_group('Regularization Parameters')
    reg_group.add_argument('--max_grad_norm', type=float, default=1.0,
                          help='梯度裁剪阈值 (防止梯度爆炸)')
    reg_group.add_argument('--feature_drop_rate', type=float, default=0.1,
                          help='训练时随机丢弃patch的比率 (数据增强)')
    reg_group.add_argument('--label_smoothing', type=float, default=0.0,
                          help='标签平滑系数 (0-0.1)')
    
    # ========== 🔥 新增: 学习率调度参数 ==========
    scheduler_group = parser.add_argument_group('Learning Rate Scheduler Parameters')
    scheduler_group.add_argument('--scheduler', type=str, default='cosine',
                                choices=['none', 'cosine', 'step', 'plateau'],
                                help='学习率调度器类型')
    scheduler_group.add_argument('--lr_step_size', type=int, default=30,
                                help='StepLR的步长 (仅scheduler=step时使用)')
    scheduler_group.add_argument('--lr_gamma', type=float, default=0.5,
                                help='StepLR的衰减率 (仅scheduler=step时使用)')
    scheduler_group.add_argument('--warmup_epochs', type=int, default=0,
                                help='学习率预热轮数 (0表示不使用)')
    
    # ========== Ranking Loss参数 ==========
    ranking_group = parser.add_argument_group('Ranking Loss Parameters')
    ranking_group.add_argument('--ranking_weight', type=float, default=0.1,
                              help='Ranking loss权重 (仅loss=combined时使用)')
    ranking_group.add_argument('--ranking_margin', type=float, default=0.0,
                              help='Ranking loss边界值 (仅loss=combined时使用)')
    
    # ========== 早停参数 ==========
    early_stop_group = parser.add_argument_group('Early Stopping Parameters')
    early_stop_group.add_argument('--warmup', type=int, default=5,
                                  help='早停预热轮数')
    early_stop_group.add_argument('--patience', type=int, default=15,
                                  help='早停耐心值')
    early_stop_group.add_argument('--stop_epoch', type=int, default=20,
                                  help='早停最小轮数')
    early_stop_group.add_argument('--early_stop_delta', type=float, default=0.0001,
                                  help='早停最小改进阈值')
    early_stop_group.add_argument('--save_all_checkpoints', action='store_true',
                                  help='是否保存所有epoch的检查点')
    
    # ========== K-Fold参数 ==========
    kfold_group = parser.add_argument_group('K-Fold Parameters')
    kfold_group.add_argument('--k_fold', type=int, default=5,
                            help='K-Fold折数')
    kfold_group.add_argument('--fold', type=int, default=None,
                            help='指定训练某个fold (None表示训练所有fold)')
    kfold_group.add_argument('--val_ratio', type=float, default=0.15,
                            help='验证集比例')
    kfold_group.add_argument('--test_ratio', type=float, default=0.15,
                            help='测试集比例')
    
    # ========== 🔥 新增: DDP参数 ==========
    ddp_group = parser.add_argument_group('Distributed Training Parameters')
    ddp_group.add_argument('--local_rank', type=int, default=-1,
                          help='DDP local rank (自动设置)')
    
    # ========== 其他参数 ==========
    misc_group = parser.add_argument_group('Miscellaneous Parameters')
    misc_group.add_argument('--results_dir', type=str, default='./results',
                           help='结果保存目录')
    misc_group.add_argument('--exp_name', type=str, default=None,
                           help='实验名称 (用于区分不同实验)')
    misc_group.add_argument('--num_workers', type=int, default=4,
                           help='数据加载线程数')
    misc_group.add_argument('--seed', type=int, default=42,
                           help='随机种子')
    misc_group.add_argument('--resume', type=str, default=None,
                           help='恢复训练的检查点路径')
    misc_group.add_argument('--eval_only', action='store_true',
                           help='仅评估模式 (需要--resume)')
    
    # ========== 解析参数 ==========
    args = parser.parse_args()
    
    # ========== 🔥 自动生成实验名称 ==========
    if args.exp_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.exp_name = f'{args.model_version}_layer{args.mamba_layer}_drop{args.dropout}_{timestamp}'
    
    # 更新结果目录
    args.results_dir = os.path.join(args.results_dir, args.exp_name)
    
    # ========== 验证参数 ==========
    # 外部测试集
    if args.external_csv_path and not args.external_h5_base_dir:
        parser.error('--external_h5_base_dir is required when --external_csv_path is provided')
    if args.external_h5_base_dir and not args.external_csv_path:
        parser.error('--external_csv_path is required when --external_h5_base_dir is provided')
    
    # 损失函数
    if args.loss == 'combined':
        if args.ranking_weight <= 0:
            print(f'⚠️  Warning: loss=combined but ranking_weight={args.ranking_weight}, setting to 0.1')
            args.ranking_weight = 0.1
    
    # 模型版本
    if args.model_version == 'lite':
        if args.mamba_layer > 1:
            print(f'⚠️  Warning: lite version with mamba_layer={args.mamba_layer}, setting to 1')
            args.mamba_layer = 1
    
    # 评估模式
    if args.eval_only and not args.resume:
        parser.error('--resume is required when --eval_only is set')
    
    # ========== 设置随机种子 ==========
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # ========== 创建结果目录 ==========
    os.makedirs(args.results_dir, exist_ok=True)
    
    # ========== 保存配置 (JSON + TXT) ==========
    # JSON格式 (方便程序读取)
    config_json_path = os.path.join(args.results_dir, 'config.json')
    with open(config_json_path, 'w') as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)
    
    # TXT格式 (方便人类阅读)
    config_txt_path = os.path.join(args.results_dir, 'config.txt')
    with open(config_txt_path, 'w') as f:
        f.write('='*80 + '\n')
        f.write(f'Experiment: {args.exp_name}\n')
        f.write(f'Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write('='*80 + '\n\n')
        
        # 按组打印
        groups = {
            'Data': ['csv_path', 'h5_base_dir', 'feature_models', 'label_col'],
            'External Test': ['external_csv_path', 'external_h5_base_dir'],
            'Model': ['model_version', 'in_dim', 'n_classes', 'dropout', 
                     'drop_path_rate', 'feature_dropout', 'act', 'mamba_layer'],
            'Training': ['batch_size', 'max_epochs', 'lr', 'weight_decay', 
                        'optimizer', 'loss', 'alpha_surv', 'gc'],
            'Regularization': ['max_grad_norm', 'feature_drop_rate', 'label_smoothing'],
            'Scheduler': ['scheduler', 'lr_step_size', 'lr_gamma', 'warmup_epochs'],
            'Ranking Loss': ['ranking_weight', 'ranking_margin'],
            'Early Stopping': ['warmup', 'patience', 'stop_epoch', 
                              'early_stop_delta', 'save_all_checkpoints'],
            'K-Fold': ['k_fold', 'fold', 'val_ratio', 'test_ratio'],
            'Misc': ['results_dir', 'exp_name', 'num_workers', 'seed', 
                    'resume', 'eval_only']
        }
        
        for group_name, keys in groups.items():
            f.write(f'\n[{group_name}]\n')
            f.write('-' * 80 + '\n')
            for key in keys:
                if key in vars(args):
                    value = getattr(args, key)
                    f.write(f'{key:25s}: {value}\n')
        
        f.write('\n' + '='*80 + '\n')
    
    # ========== 打印配置 ==========
    print('\n' + '='*80)
    print(f'Experiment: {args.exp_name}')
    print(f'Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('='*80)
    
    # 打印关键参数
    print('\n🔧 Key Parameters:')
    print(f'  Model: {args.model_version} (layer={args.mamba_layer}, dropout={args.dropout})')
    print(f'  Training: lr={args.lr}, wd={args.weight_decay}, batch={args.batch_size}')
    print(f'  Loss: {args.loss}', end='')
    if args.loss == 'combined':
        print(f' (ranking_weight={args.ranking_weight})')
    else:
        print()
    print(f'  Regularization: grad_clip={args.max_grad_norm}, feature_drop={args.feature_drop_rate}')
    print(f'  Scheduler: {args.scheduler}')
    print(f'  Early Stop: patience={args.patience}, delta={args.early_stop_delta}')
    
    # ========== 打印设备信息 ==========
    print('\n💻 Device Information:')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  Device: {device}')
    if torch.cuda.is_available():
        print(f'  GPU Count: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
            print(f'    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB')
    
    # 检查DDP
    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        print(f'\n🚀 Distributed Training:')
        print(f'  World Size: {world_size}')
        print(f'  Rank: {rank}')
    
    print('='*80 + '\n')
    
    # ========== 🔥 推荐配置检查 ==========
    warnings = []
    
    # 检查过拟合风险
    if args.dropout < 0.3:
        warnings.append(f'⚠️  dropout={args.dropout} 可能太小，推荐0.3-0.5')
    
    if args.weight_decay < 1e-4:
        warnings.append(f'⚠️  weight_decay={args.weight_decay} 可能太小，推荐1e-4到1e-2')
    
    if args.mamba_layer > 2:
        warnings.append(f'⚠️  mamba_layer={args.mamba_layer} 可能导致过拟合，推荐1-2层')
    
    if args.lr > 1e-4:
        warnings.append(f'⚠️  lr={args.lr} 可能太大，推荐1e-5到1e-4')
    
    if args.scheduler == 'none':
        warnings.append('⚠️  未使用学习率调度器，推荐使用cosine或plateau')
    
    if warnings:
        print('📋 Configuration Warnings:')
        for w in warnings:
            print(f'  {w}')
        print()
    
    # ========== 开始训练 ==========
    try:
        if args.eval_only:
            # 仅评估模式
            print(f'\n{"="*80}')
            print('评估模式 (Evaluation Only)')
            print(f'{"="*80}\n')
            
            # TODO: 实现评估函数
            print('⚠️  评估模式尚未实现')
            
        elif args.fold is not None:
            # 训练单个fold
            print(f'\n{"="*80}')
            print(f'训练单个 Fold {args.fold}')
            print(f'{"="*80}\n')
            results = train_survival(args)
            
            # 打印结果
            print(f'\n{"="*80}')
            print(f'✅ Fold {args.fold} 训练完成!')
            print(f'{"="*80}')
            if results:
                print(f"\n📊 Results:")
                print(f"  Best Val C-Index: {results.get('best_val_cindex', 0):.4f}")
                print(f"  Final Val C-Index: {results.get('val_cindex', 0):.4f}")
                print(f"  Test C-Index: {results.get('test_cindex', 0):.4f}")
                if results.get('external_cindex') is not None:
                    print(f"  External C-Index: {results['external_cindex']:.4f}")
            print()
            
        else:
            # K-Fold交叉验证
            print(f'\n{"="*80}')
            print(f'开始 {args.k_fold}-Fold 交叉验证')
            print(f'{"="*80}\n')
            summary = train_k_fold(args)
            
            # 打印汇总结果
            print(f'\n{"="*80}')
            print(f'✅ {args.k_fold}-Fold 交叉验证完成!')
            print(f'{"="*80}')
            if summary:
                print(f"\n📊 Summary:")
                print(f"  Val C-Index: {summary.get('mean_val_cindex', 0):.4f} ± {summary.get('std_val_cindex', 0):.4f}")
                print(f"  Test C-Index: {summary.get('mean_test_cindex', 0):.4f} ± {summary.get('std_test_cindex', 0):.4f}")
                if summary.get('mean_external_cindex') is not None:
                    print(f"  External C-Index: {summary['mean_external_cindex']:.4f} ± {summary['std_external_cindex']:.4f}")
                
                # 打印每个fold的结果
                print(f"\n📋 Per-Fold Results:")
                for i in range(args.k_fold):
                    print(f"  Fold {i}: Val={summary['val_cindices'][i]:.4f}, Test={summary['test_cindices'][i]:.4f}", end='')
                    if summary.get('external_cindices'):
                        print(f", External={summary['external_cindices'][i]:.4f}")
                    else:
                        print()
            print()
            
    except KeyboardInterrupt:
        print('\n\n❌ 训练被用户中断!')
        sys.exit(0)
    except Exception as e:
        print(f'\n\n❌ 训练过程中发生错误: {str(e)}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print('='*80)
    print('🎉 训练完成!')
    print(f'📁 结果保存在: {args.results_dir}')
    print('='*80)


if __name__ == '__main__':
    main()
