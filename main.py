"""
生存分析训练主脚本
"""
import os
import sys
import argparse
import torch
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.core_utils import train_survival, train_k_fold


def main():
    parser = argparse.ArgumentParser(description='Survival Analysis Training')
    
    # ========== 数据参数 ==========
    parser.add_argument('--csv_path', type=str, required=True,
                       help='训练集CSV文件路径')
    parser.add_argument('--h5_dir', type=str, required=True,
                       help='训练集H5特征文件目录')
    parser.add_argument('--label_col', type=str, default='disc_label',
                       help='标签列名')
    
    # ========== 外部测试集参数 ==========
    parser.add_argument('--external_csv_path', type=str, default=None,
                       help='外部测试集CSV文件路径 (可选)')
    parser.add_argument('--external_h5_dir', type=str, default=None,
                       help='外部测试集H5特征文件目录 (可选)')
    
    # ========== 模型参数 ==========
    parser.add_argument('--in_dim', type=int, default=512,
                       help='输入特征维度')
    parser.add_argument('--n_classes', type=int, default=4,
                       help='生存时间区间数量')
    parser.add_argument('--dropout', type=float, default=0.25,
                       help='Dropout比率')
    parser.add_argument('--act', type=str, default='gelu',
                       choices=['relu', 'gelu'],
                       help='激活函数')
    parser.add_argument('--mamba_layer', type=int, default=2,
                       help='Mamba层数')
    
    # ========== 训练参数 ==========
    parser.add_argument('--batch_size', type=int, default=1,
                       help='批大小 (MIL任务通常使用1)')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='最大训练轮数')
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='权重衰减')
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw'],
                       help='优化器')
    parser.add_argument('--loss', type=str, default='nll',
                       choices=['cox', 'nll', 'combined'],
                       help='损失函数类型')
    parser.add_argument('--alpha_surv', type=float, default=0.0,
                       help='NLL损失的alpha参数')
    
    # ========== Ranking Loss参数 ==========
    parser.add_argument('--ranking_weight', type=float, default=0.1,
                       help='Ranking loss权重 (仅当loss=combined时使用)')
    parser.add_argument('--ranking_margin', type=float, default=0.0,
                       help='Ranking loss边界值 (仅当loss=combined时使用)')
    parser.add_argument('--gc', type=int, default=1,
                       help='梯度累积步数')
    
    # ========== 早停参数 ==========
    parser.add_argument('--warmup', type=int, default=5,
                       help='早停预热轮数')
    parser.add_argument('--patience', type=int, default=15,
                       help='早停耐心值')
    parser.add_argument('--stop_epoch', type=int, default=20,
                       help='早停最小轮数')
    
    # ========== K-Fold参数 ==========
    parser.add_argument('--k_fold', type=int, default=5,
                       help='K-Fold折数')
    parser.add_argument('--fold', type=int, default=None,
                       help='指定训练某个fold (None表示训练所有fold)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='测试集比例')
    
    # ========== 其他参数 ==========
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='结果保存目录')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='数据加载线程数')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    # ========== 解析参数 ==========
    args = parser.parse_args()
    
    # ========== 验证外部测试集参数 ==========
    if args.external_csv_path and not args.external_h5_dir:
        parser.error('--external_h5_dir is required when --external_csv_path is provided')
    if args.external_h5_dir and not args.external_csv_path:
        parser.error('--external_csv_path is required when --external_h5_dir is provided')
    
    # ========== 验证损失函数参数 ==========
    if args.loss == 'combined':
        if args.ranking_weight <= 0:
            print(f'⚠️  Warning: loss=combined but ranking_weight={args.ranking_weight}, setting to 0.1')
            args.ranking_weight = 0.1
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 创建结果目录
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 保存配置
    config_path = os.path.join(args.results_dir, 'config.txt')
    with open(config_path, 'w') as f:
        f.write('='*60 + '\n')
        f.write('Training Configuration\n')
        f.write('='*60 + '\n')
        for key, value in sorted(vars(args).items()):
            f.write(f'{key}: {value}\n')
        f.write('='*60 + '\n')
    
    # 打印配置
    print('\n' + '='*60)
    print('Training Configuration')
    print('='*60)
    for key, value in sorted(vars(args).items()):
        print(f'{key}: {value}')
    print('='*60 + '\n')
    
    # 打印设备信息
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
    print()
    
    # 开始训练
    try:
        if args.fold is not None:
            # 训练单个fold
            print(f'\n{"="*60}')
            print(f'训练单个 Fold {args.fold}')
            print(f'{"="*60}\n')
            results = train_survival(args)
            
            # 打印结果
            print(f'\n{"="*60}')
            print(f'Fold {args.fold} 训练完成!')
            print(f'{"="*60}')
            if results:
                print(f"\n最佳验证 C-Index: {results.get('best_val_cindex', 0):.4f}")
                print(f"测试 C-Index: {results.get('test_cindex', 0):.4f}")
                if results.get('external_cindex') is not None:
                    print(f"外部测试 C-Index: {results['external_cindex']:.4f}")
            print()
            
        else:
            # K-Fold交叉验证
            print(f'\n{"="*60}')
            print(f'开始 {args.k_fold}-Fold 交叉验证')
            print(f'{"="*60}\n')
            summary = train_k_fold(args)
            
            # 打印汇总结果
            print(f'\n{"="*60}')
            print(f'{args.k_fold}-Fold 交叉验证完成!')
            print(f'{"="*60}')
            if summary:
                print(f"\n平均验证 C-Index: {summary.get('mean_val_cindex', 0):.4f} ± {summary.get('std_val_cindex', 0):.4f}")
                print(f"平均测试 C-Index: {summary.get('mean_test_cindex', 0):.4f} ± {summary.get('std_test_cindex', 0):.4f}")
                if summary.get('mean_external_cindex') is not None:
                    print(f"平均外部测试 C-Index: {summary['mean_external_cindex']:.4f} ± {summary['std_external_cindex']:.4f}")
            print()
            
    except KeyboardInterrupt:
        print('\n\n训练被用户中断!')
        sys.exit(0)
    except Exception as e:
        print(f'\n\n训练过程中发生错误: {str(e)}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print('='*60)
    print('训练完成!')
    print('='*60)


if __name__ == '__main__':
    main()
