# train_ddp.py (增强版)
"""
DDP训练入口 - 用于Optuna超参数优化
调用 core_utils2.py 的 train_survival() 函数

增强功能:
- 启动时自动进行DDP环境和NCCL配置自检，方便调试。
- 优化了参数处理和日志输出。
"""

import argparse
import os
import sys
from argparse import Namespace

def setup_ddp_and_print_env():
    """
    🔥 DDP环境自检与日志打印 (核心增强功能)
    - 检查torchrun设置的环境变量，确认DDP状态。
    - 打印关键的NCCL配置，用于快速诊断通信问题。
    """
    print("\n" + "="*60)
    print("🔍 DDP 环境与 NCCL 配置自检")
    print("="*60)
    
    rank = os.environ.get('RANK', '未设置')
    local_rank = os.environ.get('LOCAL_RANK', '未设置')
    world_size = os.environ.get('WORLD_SIZE', '未设置')

    if '未设置' in [rank, local_rank, world_size]:
        print("⚠️  警告: 未检测到DDP环境变量 (RANK, LOCAL_RANK, WORLD_SIZE)。")
        print("   脚本可能未通过 torchrun 或类似工具启动，将以单进程模式运行。")
    else:
        print(f"✓ DDP环境已激活:")
        print(f"  - 全局进程ID (RANK): {rank}")
        print(f"  - 本地GPU ID (LOCAL_RANK): {local_rank}")
        print(f"  - 总进程数 (WORLD_SIZE): {world_size}")

    print("\n--- NCCL 配置 ---")
    nccl_vars = {
        'NCCL_SOCKET_IFNAME': '网络接口',
        'NCCL_IB_DISABLE': '禁用InfiniBand',
        'NCCL_P2P_DISABLE': '禁用GPU点对点',
        'NCCL_SHM_DISABLE': '禁用共享内存',
        'NCCL_BLOCKING_WAIT': '阻塞等待模式',
        'NCCL_DEBUG': '调试级别',
        'NCCL_TIMEOUT': '超时时间(秒)'
    }
    for var, desc in nccl_vars.items():
        value = os.environ.get(var, '未设置')
        print(f"  - {desc} ({var}): {value}")
    
    print("="*60)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Survival Analysis Training with DDP',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # 默认值也会显示在help信息里
    )
    
    # ==================== 数据路径 ====================
    group = parser.add_argument_group('数据路径参数')
    group.add_argument('--csv_path', type=str, required=True, help='训练集CSV路径')
    group.add_argument('--h5_dir', type=str, required=True, help='H5特征文件目录')
    group.add_argument('--external_csv_path', type=str, default=None, help='外部测试集CSV路径')
    group.add_argument('--external_h5_dir', type=str, default=None, help='外部测试集H5目录')
    
    # ==================== 模型参数 ====================
    group = parser.add_argument_group('模型结构参数')
    group.add_argument('--in_dim', type=int, default=768, help='输入特征维度')
    group.add_argument('--n_classes', type=int, default=4, help='生存分析离散化类别数')
    group.add_argument('--dropout', type=float, default=0.25, help='Dropout比例')
    group.add_argument('--act', type=str, default='gelu', choices=['relu', 'gelu', 'silu'], help='激活函数')
    group.add_argument('--mamba_layer', type=int, default=2, help='Mamba2层数')
    group.add_argument('--use_clinical', action='store_true', help='是否使用临床特征(性别、年龄)')
    
    # ==================== 特征模型 ====================
    group = parser.add_argument_group('特征融合参数')
    group.add_argument('--feature_models', type=str, nargs='+', default=['ctranspath'], help='使用的特征模型列表 (例如: ctranspath uni_v1)')
    
    # ==================== 训练参数 ====================
    group = parser.add_argument_group('核心训练参数')
    group.add_argument('--batch_size', type=int, default=4, help='每个GPU的批次大小')
    group.add_argument('--max_epochs', type=int, default=100, help='最大训练轮数')
    group.add_argument('--lr', type=float, default=2e-4, help='学习率')
    group.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    group.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'], help='优化器')
    group.add_argument('--gc', type=int, default=1, help='梯度累积步数')
    group.add_argument('--max_grad_norm', type=float, default=1.0, help='梯度裁剪阈值')
    
    # ==================== 损失函数 ====================
    group = parser.add_argument_group('损失函数参数')
    group.add_argument('--loss', type=str, default='nll', choices=['nll', 'cox', 'combined'], help='损失函数类型')
    group.add_argument('--main_loss_type', type=str, default='nll', choices=['nll', 'cox'], help='当loss=combined时, 指定主损失')
    group.add_argument('--alpha_surv', type=float, default=0.15, help='NLL损失的alpha平滑参数')
    group.add_argument('--ranking_weight', type=float, default=0.0, help='Ranking损失的权重 (当loss=combined时生效)')
    group.add_argument('--ranking_margin', type=float, default=0.0, help='Ranking损失的边界')
    
    # ==================== 学习率调度器 ====================
    group = parser.add_argument_group('学习率调度器参数')
    group.add_argument('--scheduler', type=str, default=None, choices=['cosine', 'step', 'plateau'], help='学习率调度器类型')
    group.add_argument('--min_lr', type=float, default=1e-6, help='[Cosine] 最小学习率')
    group.add_argument('--step_size', type=int, default=30, help='[Step] 步长 (兼容旧版lr_step_size)')
    group.add_argument('--gamma', type=float, default=0.1, help='[Step] 衰减率 (兼容旧版lr_gamma)')
    
    # ==================== 数据集划分 ====================
    group = parser.add_argument_group('数据集划分参数')
    group.add_argument('--k_fold', type=int, default=10, help='K-fold交叉验证的折数')
    group.add_argument('--fold', type=int, default=0, help='当前训练的fold索引 (0-based)')
    group.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例 (当k_fold=1时生效)')
    group.add_argument('--test_ratio', type=float, default=0.1, help='测试集比例 (当k_fold=1时生效)')
    
    # ==================== 早停策略 ====================
    group = parser.add_argument_group('早停策略参数')
    group.add_argument('--warmup', type=int, default=5, help='早停预热轮数 (此期间不触发早停)')
    group.add_argument('--patience', type=int, default=15, help='早停耐心值')
    group.add_argument('--stop_epoch', type=int, default=30, help='最早允许停止的轮数')
    group.add_argument('--early_stop_delta', type=float, default=0.0001, help='早停改进的最小阈值')
    
    # ==================== 其他 ====================
    group = parser.add_argument_group('其他参数')
    group.add_argument('--results_dir', type=str, default='./results', help='结果保存目录')
    group.add_argument('--num_workers', type=int, default=4, help='数据加载的线程数')
    group.add_argument('--seed', type=int, default=42, help='全局随机种子')
    group.add_argument('--label_col', type=str, default='disc_label', help='生存分析标签列名')
    group.add_argument('--normalize_age', action='store_true', default=True, help='是否标准化年龄特征')
    group.add_argument('--save_all_checkpoints', action='store_true', help='是否保存所有轮次的模型权重')
    
    return parser.parse_args()


def process_args(args):
    """
    处理参数 (转换为 core_utils2.py 期望的格式)
    """
    # 处理 feature_models: 如果只有一个模型，从列表转为字符串
    if args.feature_models and len(args.feature_models) == 1:
        args.feature_models = args.feature_models[0]
    
    # 兼容旧版调度器参数名
    args.lr_step_size = args.step_size
    args.lr_gamma = args.gamma
    
    # 兼容旧版h5目录参数名
    args.h5_base_dir = args.h5_dir
    if args.external_h5_dir:
        args.external_h5_base_dir = args.external_h5_dir
    
    return args


def main():
    """主函数"""
    # 1. DDP环境自检 (增强功能)
    setup_ddp_and_print_env()
    
    # 2. 解析和处理参数
    args = parse_args()
    args = process_args(args)
    
    # 3. 打印本次运行的核心配置
    # (只在主进程打印，避免DDP多进程重复输出)
    if os.environ.get('RANK', '0') == '0':
        print("\n" + "="*60)
        print("🚀 训练核心配置")
        print("="*60)
        print(f"  - 数据集: {args.csv_path}")
        print(f"  - H5目录: {args.h5_base_dir}")
        print(f"  - 特征模型: {args.feature_models}")
        print(f"  - Fold: {args.fold + 1}/{args.k_fold}") # 改为 1-based 更直观
        print(f"  - 批次大小 (per GPU): {args.batch_size}")
        print(f"  - 学习率: {args.lr}")
        print(f"  - 优化器: {args.optimizer}")
        print(f"  - 损失函数: {args.loss}")
        if args.loss == 'combined':
            print(f"    - 主损失: {args.main_loss_type}, Alpha: {args.alpha_surv}, Ranking权重: {args.ranking_weight}")
        print(f"  - 结果目录: {args.results_dir}")
        print("="*60 + "\n")
    
    # 4. 动态导入核心训练函数
    try:
        from utils.core_utils2 import train_survival
    except ImportError as e:
        print(f"❌ 致命错误: 导入 'train_survival' 失败: {e}", file=sys.stderr)
        print("   请确保 'utils/core_utils2.py' 文件存在且路径正确。", file=sys.stderr)
        sys.exit(1)
    
    # 5. 开始训练
    try:
        results = train_survival(args)
        
        # 只在主进程打印最终结果
        if os.environ.get('RANK', '0') == '0':
            if results:
                print("\n" + "="*60)
                print("✅ 训练顺利完成!")
                print("="*60)
                print(f"  - 验证集 C-Index: {results.get('val_cindex', 'N/A'):.4f}")
                print(f"  - 测试集 C-Index: {results.get('test_cindex', 'N/A'):.4f}")
                if 'external_cindex' in results:
                    print(f"  - 外部测试集 C-Index: {results.get('external_cindex', 'N/A'):.4f}")
                print("="*60)
            else:
                print("\n❌ 训练失败: 训练函数返回了 None。", file=sys.stderr)
                sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ 训练过程中发生致命异常: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
