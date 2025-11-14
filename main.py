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
                       choices=['cox', 'nll'],
                       help='损失函数类型')
    parser.add_argument('--alpha_surv', type=float, default=0.0,
                       help='NLL损失的alpha参数')
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
    
    # ========== 解析参数 (必须在这里!) ==========
    args = parser.parse_args()
    
    # ========== 验证外部测试集参数 ==========
    if args.external_csv_path and not args.external_h5_dir:
        parser.error('--external_h5_dir is required when --external_csv_path is provided')
    if args.external_h5_dir and not args.external_csv_path:
        parser.error('--external_csv_path is required when --external_h5_dir is provided')
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
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
    
    print('\n' + '='*60)
    print('Training Configuration')
    print('='*60)
    for key, value in sorted(vars(args).items()):
        print(f'{key}: {value}')
    print('='*60 + '\n')
    
    # 开始训练
    if args.fold is not None:
        # 训练单个fold
        print(f'\n训练单个 Fold {args.fold}...\n')
        results = train_survival(args)
    else:
        # K-Fold交叉验证
        print(f'\n开始 {args.k_fold}-Fold 交叉验证...\n')
        summary = train_k_fold(args)


if __name__ == '__main__':
    main()