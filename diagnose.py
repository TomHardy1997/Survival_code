"""
诊断脚本 - 找出卡住的位置
"""

import os
import subprocess
import time

# 配置
CSV_PATH = "/home/stat-jijianxin/PFMs/Survival_code/csv_file/hmu_survival_with_slides.csv"
H5_DIR = "/home/stat-jijianxin/PFMs/HMU_GC_ALL_H5"

def test_data_loading():
    """测试1: 数据加载"""
    print("\n" + "="*60)
    print("🧪 测试1: 数据加载")
    print("="*60)
    
    test_code = f"""
import sys
sys.path.append('.')

print("导入库...")
from dataset.dataset_xiugai import Generic_MIL_Survival_Dataset
import pandas as pd

print("读取CSV...")
df = pd.read_csv("{CSV_PATH}")
print(f"  样本数: {{len(df)}}")

print("\\n创建数据集...")
dataset = Generic_MIL_Survival_Dataset(
    csv_path="{CSV_PATH}",
    h5_dir="{H5_DIR}",
    feature_models=['ctranspath'],
    shuffle=False,
    seed=42,
    print_info=True,
    n_bins=4,
    label_col='survival_months',
    ignore_missing=True
)

print(f"\\n数据集大小: {{len(dataset)}}")

print("\\n测试加载前3个样本...")
for i in range(min(3, len(dataset))):
    print(f"  加载样本 {{i}}...", end='')
    try:
        data = dataset[i]
        print(f" ✓ (features shape: {{data['features'].shape}})")
    except Exception as e:
        print(f" ✗ 错误: {{e}}")
        import traceback
        traceback.print_exc()

print("\\n✅ 数据加载测试完成!")
"""
    
    with open('/tmp/test_data.py', 'w') as f:
        f.write(test_code)
    
    try:
        result = subprocess.run(
            ['python', '/tmp/test_data.py'],
            timeout=60,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print("错误:", result.stderr)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("❌ 数据加载超时!")
        return False
    except Exception as e:
        print(f"❌ 异常: {e}")
        return False


def test_model_forward():
    """测试2: 模型前向传播"""
    print("\n" + "="*60)
    print("🧪 测试2: 模型前向传播")
    print("="*60)
    
    test_code = """
import torch
import sys
sys.path.append('.')

print("导入模型...")
from models.Mamba2MIL2 import Mamba2MIL

print("创建模型...")
model = Mamba2MIL(
    in_dim=768,
    n_classes=4,
    dropout=0.25,
    act='relu',
    mamba_layer=2
)

print("移动到GPU...")
device = torch.device('cuda:0')
model = model.to(device)

print("\\n创建测试数据...")
batch_size = 2
seq_len = 100
x = torch.randn(batch_size, seq_len, 768).to(device)

print("前向传播...")
with torch.no_grad():
    hazards, S, Y_hat, A = model(x)

print(f"  hazards shape: {hazards.shape}")
print(f"  S shape: {S.shape}")
print(f"  Y_hat shape: {Y_hat.shape}")
print(f"  A shape: {A.shape}")

print("\\n✅ 模型前向传播测试完成!")
"""
    
    with open('/tmp/test_model.py', 'w') as f:
        f.write(test_code)
    
    try:
        result = subprocess.run(
            ['python', '/tmp/test_model.py'],
            timeout=60,
            capture_output=True,
            text=True,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0'}
        )
        print(result.stdout)
        if result.stderr:
            print("错误:", result.stderr)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("❌ 模型前向传播超时!")
        return False
    except Exception as e:
        print(f"❌ 异常: {e}")
        return False


def test_ddp_init():
    """测试3: DDP初始化"""
    print("\n" + "="*60)
    print("🧪 测试3: DDP初始化")
    print("="*60)
    
    test_code = """
import os
import torch
import torch.distributed as dist

print("设置环境变量...")
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29600'
os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
os.environ['NCCL_IB_DISABLE'] = '1'
os.environ['NCCL_P2P_DISABLE'] = '0'

rank = int(os.environ.get('RANK', 0))
world_size = int(os.environ.get('WORLD_SIZE', 1))
local_rank = int(os.environ.get('LOCAL_RANK', 0))

print(f"Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")

print("\\n初始化进程组...")
dist.init_process_group(
    backend='nccl',
    init_method='env://',
    world_size=world_size,
    rank=rank
)

print("设置设备...")
torch.cuda.set_device(local_rank)
device = torch.device(f'cuda:{local_rank}')

print("\\n测试通信...")
tensor = torch.ones(1).to(device) * rank
print(f"  Rank {rank} 发送: {tensor.item()}")

dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
print(f"  Rank {rank} 接收: {tensor.item()}")

print("\\n清理...")
dist.destroy_process_group()

print("\\n✅ DDP初始化测试完成!")
"""
    
    with open('/tmp/test_ddp.py', 'w') as f:
        f.write(test_code)
    
    try:
        result = subprocess.run(
            [
                'torchrun',
                '--nproc_per_node=2',
                '--master_addr=127.0.0.1',
                '--master_port=29600',
                '/tmp/test_ddp.py'
            ],
            timeout=60,
            capture_output=True,
            text=True,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0,1'}
        )
        print(result.stdout)
        if result.stderr:
            print("错误:", result.stderr)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("❌ DDP初始化超时!")
        return False
    except Exception as e:
        print(f"❌ 异常: {e}")
        return False
    finally:
        # 清理
        subprocess.run("pkill -9 -f 'test_ddp.py' 2>/dev/null || true", shell=True)
        time.sleep(2)


def test_simple_training():
    """测试4: 最简单的训练循环"""
    print("\n" + "="*60)
    print("🧪 测试4: 简单训练循环 (单GPU)")
    print("="*60)
    
    test_code = f"""
import torch
import sys
sys.path.append('.')

print("导入...")
from dataset.dataset_xiugai import Generic_MIL_Survival_Dataset
from models.Mamba2MIL2 import Mamba2MIL
from torch.utils.data import DataLoader

print("\\n创建数据集...")
dataset = Generic_MIL_Survival_Dataset(
    csv_path="{CSV_PATH}",
    h5_dir="{H5_DIR}",
    feature_models=['ctranspath'],
    shuffle=False,
    seed=42,
    print_info=False,
    n_bins=4,
    label_col='survival_months',
    ignore_missing=True
)

print(f"数据集大小: {{len(dataset)}}")

print("\\n创建DataLoader...")
loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0,  # 单进程
    pin_memory=False
)

print("\\n创建模型...")
model = Mamba2MIL(
    in_dim=768,
    n_classes=4,
    dropout=0.25,
    act='relu',
    mamba_layer=2
).cuda()

print("\\n测试训练循环 (3个batch)...")
model.train()
for i, batch in enumerate(loader):
    if i >= 3:
        break
    
    print(f"  Batch {{i+1}}/3...", end='')
    
    features = batch['features'].cuda()
    label = batch['label'].cuda()
    
    # 前向
    hazards, S, Y_hat, A = model(features)
    
    # 简单损失
    loss = torch.nn.functional.cross_entropy(hazards, label)
    
    # 反向
    loss.backward()
    
    print(f" ✓ (loss: {{loss.item():.4f}})")

print("\\n✅ 简单训练循环测试完成!")
"""
    
    with open('/tmp/test_train.py', 'w') as f:
        f.write(test_code)
    
    try:
        result = subprocess.run(
            ['python', '/tmp/test_train.py'],
            timeout=120,
            capture_output=True,
            text=True,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0'}
        )
        print(result.stdout)
        if result.stderr:
            print("错误:", result.stderr)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("❌ 训练循环超时!")
        return False
    except Exception as e:
        print(f"❌ 异常: {e}")
        return False


def main():
    print("\n" + "="*70)
    print(" "*20 + "🔍 诊断脚本")
    print("="*70)
    
    tests = [
        ("数据加载", test_data_loading),
        ("模型前向传播", test_model_forward),
        ("DDP初始化", test_ddp_init),
        ("简单训练循环", test_simple_training),
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ {name} 测试异常: {e}")
            results[name] = False
        
        time.sleep(2)
    
    # 总结
    print("\n" + "="*70)
    print(" "*25 + "📊 诊断结果")
    print("="*70)
    
    for name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {name:20s}: {status}")
    
    print("="*70)
    
    # 分析
    print("\n💡 分析:")
    if not results.get("数据加载", False):
        print("  ⚠️  数据加载有问题，检查:")
        print("     - CSV文件格式")
        print("     - H5文件路径")
        print("     - 特征文件是否存在")
    
    if not results.get("模型前向传播", False):
        print("  ⚠️  模型有问题，检查:")
        print("     - Mamba2MIL2.py 实现")
        print("     - GPU内存")
    
    if not results.get("DDP初始化", False):
        print("  ⚠️  DDP初始化有问题，检查:")
        print("     - NCCL版本")
        print("     - 网络配置")
        print("     - 防火墙设置")
    
    if not results.get("简单训练循环", False):
        print("  ⚠️  训练循环有问题，检查:")
        print("     - DataLoader配置")
        print("     - 损失函数")
        print("     - GPU内存")
    
    if all(results.values()):
        print("  🎉 所有基础测试通过!")
        print("  ➡️  问题可能在 train_ddp.py 的复杂逻辑中")


if __name__ == "__main__":
    main()
