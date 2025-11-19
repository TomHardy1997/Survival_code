"""
Optuna 超参数优化 - DDP多GPU版本 (三阶段方案) - 报错跳过版本
"""
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
import sys
import subprocess
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import time
import warnings
import signal
warnings.filterwarnings('ignore')

# ===================== 配置 =====================
STUDY_NAME = "mamba2mil_survival_ddp_optimization"
STORAGE_PATH = "./results/optuna_study/optuna.db"
N_JOBS = 1

# DDP配置
NUM_GPUS = 2
CUDA_VISIBLE_DEVICES = "0,1"
MASTER_ADDR = "127.0.0.1"
BASE_PORT = 29500

# 数据路径
CSV_PATH = "/home/stat-jijianxin/PFMs/Survival_code/csv_file/hmu_survival_with_slides.csv"
H5_DIR = "/home/stat-jijianxin/PFMs/HMU_GC_ALL_H5/"
EXTERNAL_CSV = "/home/stat-jijianxin/PFMs/Survival_code/csv_file/tcga_survival_matched.csv"
EXTERNAL_H5 = "/home/stat-jijianxin/PFMs/TRIDENT/tcga_filtered/20x_512px_0px_overlap/"

# 固定参数
FIXED_PARAMS = {
    # 数据
    'csv_path': CSV_PATH,
    'h5_dir': H5_DIR,
    'external_csv_path': EXTERNAL_CSV,
    'external_h5_dir': EXTERNAL_H5,
    
    # 模型基础
    'in_dim': 768,
    'n_classes': 4,
    'feature_models': 'ctranspath',
    
    # 训练策略
    'max_epochs': 100,
    'stop_epoch': 30,
    'warmup': 5,
    'patience': 15,
    'early_stop_delta': 0.0001,
    
    # 损失函数
    'loss': 'combined',
    'main_loss_type': 'nll',
    'alpha_surv': 0.365,
    'ranking_margin': 0.0,
    
    # 数据集划分
    'k_fold': 10,
    'val_ratio': 0.1,
    'test_ratio': 0.1,
    'seed': 123,
    'num_workers': 0,
}


# ===================== 环境配置函数 =====================
def setup_environment():
    """设置缓存路径和NCCL配置"""
    home_cache = os.path.expanduser("~/.cache")
    
    os.environ['HOME_CACHE'] = home_cache
    os.environ['TRITON_CACHE_DIR'] = f"{home_cache}/triton"
    os.environ['TORCH_COMPILE_CACHE_DIR'] = f"{home_cache}/torch_compile"
    os.environ['TRANSFORMERS_CACHE'] = f"{home_cache}/transformers"
    os.environ['HF_HOME'] = f"{home_cache}/huggingface"
    
    # 创建目录
    os.makedirs(os.environ['TRITON_CACHE_DIR'], exist_ok=True)
    os.makedirs(os.environ['TORCH_COMPILE_CACHE_DIR'], exist_ok=True)
    
    # NCCL配置
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_P2P_DISABLE'] = '0'
    os.environ['NCCL_SHM_DISABLE'] = '0'
    os.environ['NCCL_BLOCKING_WAIT'] = '1'
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
    os.environ['NCCL_DEBUG'] = 'WARN'
    os.environ['NCCL_TIMEOUT'] = '1800'
    
    print(f"✓ 缓存路径已设置到: {home_cache}")


def cleanup_resources():
    """强化资源清理"""
    print("🧹 强化资源清理...")
    
    # 1. 杀死所有相关进程
    subprocess.run("pkill -9 -f 'torchrun' 2>/dev/null || true", shell=True)
    subprocess.run("pkill -9 -f 'train_ddp.py' 2>/dev/null || true", shell=True)
    subprocess.run("pkill -9 -f 'python.*train_ddp.py' 2>/dev/null || true", shell=True)
    
    # 2. 清理CUDA缓存
    subprocess.run("""
python3 << 'PYEOF'
import torch
import gc
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
gc.collect()
PYEOF
""", shell=True)
    
    # 3. 清理共享内存
    subprocess.run("rm -rf /dev/shm/torch_* 2>/dev/null || true", shell=True)
    
    # 4. 清理Triton缓存
    triton_cache = os.environ.get('TRITON_CACHE_DIR')
    if triton_cache:
        subprocess.run(f"rm -rf {triton_cache}/* 2>/dev/null || true", shell=True)
    
    # 5. 清理/tmp临时文件
    subprocess.run("rm -rf /tmp/triton_cache_rank_* 2>/dev/null || true", shell=True)
    subprocess.run("rm -rf /tmp/torch_* 2>/dev/null || true", shell=True)
    
    time.sleep(3)
    print("✓ 资源清理完成")


def wait_for_port(port, max_wait=30):
    """等待端口释放"""
    waited = 0
    while waited < max_wait:
        result = subprocess.run(
            f"netstat -tuln 2>/dev/null | grep -q ':{port} '",
            shell=True,
            capture_output=True
        )
        if result.returncode != 0:
            return True
        
        print(f"⏳ 等待端口 {port} 释放... ({waited}/{max_wait})")
        time.sleep(1)
        waited += 1
    
    print(f"⚠️  端口 {port} 仍被占用，强制清理...")
    subprocess.run(f"fuser -k {port}/tcp 2>/dev/null || true", shell=True)
    time.sleep(2)
    return True


# ===================== DDP训练函数 - 报错跳过版本 =====================
def run_ddp_training(params, trial_number, fold=0):
    """运行DDP训练 - 报错就跳过"""
    master_port = BASE_PORT + trial_number
    
    # 🔥 任何异常都直接返回 None，让 Optuna 跳过这个 trial
    try:
        cleanup_resources()
        wait_for_port(master_port)
        
        results_dir = f"./results/optuna_study/trial_{trial_number}"
        os.makedirs(results_dir, exist_ok=True)
        
        log_file = os.path.join(results_dir, f"trial_{trial_number}_fold_{fold}.log")
        
        # 构建基础命令
        base_cmd = [
            "torchrun",
            f"--nproc_per_node={NUM_GPUS}",
            f"--master_addr={MASTER_ADDR}",
            f"--master_port={master_port}",
            "--node_rank=0",
            "--nnodes=1",
            "train_ddp.py",
            "--csv_path", params['csv_path'],
            "--h5_dir", params['h5_dir'],
            "--external_csv_path", params['external_csv_path'],
            "--external_h5_dir", params['external_h5_dir'],
            "--in_dim", str(params['in_dim']),
            "--n_classes", str(params['n_classes']),
            "--dropout", str(params['dropout']),
            "--act", params['act'],
            "--mamba_layer", str(params['mamba_layer']),
            "--batch_size", str(params['batch_size']),
            "--max_epochs", str(params['max_epochs']),
            "--lr", str(params['lr']),
            "--weight_decay", str(params['weight_decay']),
            "--optimizer", params['optimizer'],
            "--loss", params['loss'],
            "--main_loss_type", params['main_loss_type'],
            "--alpha_surv", str(params['alpha_surv']),
            "--ranking_weight", str(params['ranking_weight']),
            "--ranking_margin", str(params['ranking_margin']),
            "--gc", str(params['gc']),
            "--k_fold", str(params['k_fold']),
            "--fold", str(fold),
            "--val_ratio", str(params['val_ratio']),
            "--test_ratio", str(params['test_ratio']),
            "--warmup", str(params['warmup']),
            "--patience", str(params['patience']),
            "--stop_epoch", str(params['stop_epoch']),
            "--results_dir", results_dir,
            "--num_workers", str(params['num_workers']),
            "--seed", str(params['seed']),
        ]
        
        # 添加 feature_models
        if 'feature_models' in params:
            feature_models = params['feature_models']
            if isinstance(feature_models, list):
                for model in feature_models:
                    base_cmd.extend(["--feature_models", model])
            else:
                base_cmd.extend(["--feature_models", feature_models])
        
        # 添加调度器参数
        if 'scheduler' in params:
            base_cmd.extend(["--scheduler", params['scheduler']])
            if params['scheduler'] == 'cosine':
                base_cmd.extend(["--min_lr", str(params.get('min_lr', 1e-6))])
            elif params['scheduler'] == 'step':
                base_cmd.extend([
                    "--step_size", str(params.get('step_size', 30)),
                    "--gamma", str(params.get('gamma', 0.1))
                ])
        
        # 构建完整命令
        cmd_str = ' '.join([f'"{arg}"' if ' ' in str(arg) else str(arg) for arg in base_cmd])
        full_cmd = f'bash -c "set -o pipefail; {cmd_str} 2>&1 | tee {log_file}"'
        
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
        
        print(f"\n{'='*60}")
        print(f"🚀 启动 Trial {trial_number} (Fold {fold})")
        print(f"端口: {master_port}")
        print(f"日志: {log_file}")
        print(f"{'='*60}\n")
        
        # 启动进程
        process = subprocess.Popen(
            full_cmd,
            shell=True,
            env=env,
            stdout=sys.stdout,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid
        )
        
        print(f"✓ 进程已启动 (PID: {process.pid})")
        
        # 等待完成（4小时超时）
        try:
            returncode = process.wait(timeout=14400)
        except subprocess.TimeoutExpired:
            print(f"\n⏱️  训练超时 (4小时)，跳过此 trial")
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except:
                pass
            cleanup_resources()
            return None
        
        # 检查返回码
        if returncode != 0:
            print(f"\n❌ 训练失败 (exitcode={returncode})，跳过此 trial")
            return None
        
        # 读取结果
        results_file = os.path.join(results_dir, f'fold_{fold}', 'results.pkl')
        if not os.path.exists(results_file):
            print(f"\n❌ 结果文件不存在，跳过此 trial")
            return None
        
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        
        val_cindex = results.get('val_cindex')
        print(f"\n{'='*60}")
        print(f"✅ Trial {trial_number} 完成")
        print(f"   验证集 C-index: {val_cindex:.4f}")
        print(f"{'='*60}\n")
        
        cleanup_resources()
        time.sleep(5)
        
        return val_cindex
    
    except Exception as e:
        # 🔥 任何异常都打印后跳过
        print(f"\n⚠️  Trial {trial_number} 发生异常，跳过: {e}")
        
        # 清理进程
        try:
            subprocess.run("pkill -9 -f 'torchrun' 2>/dev/null || true", shell=True)
            subprocess.run("pkill -9 -f 'train_ddp.py' 2>/dev/null || true", shell=True)
        except:
            pass
        
        cleanup_resources()
        return None


# ===================== Optuna目标函数 =====================
def objective_stage1(trial):
    """阶段1: 核心架构"""
    params = FIXED_PARAMS.copy()
    
    params.update({
        # 🔥 dropout 改为离散选择，范围调高
        'dropout': trial.suggest_categorical('dropout', [0.6, 0.7, 0.8]),
        
        'act': trial.suggest_categorical('act', ['relu', 'gelu']),
        
        # 🔥 mamba_layer 本来就是离散的
        'mamba_layer': trial.suggest_int('mamba_layer', 1, 4),
        
        'batch_size': trial.suggest_categorical('batch_size', [4, 8, 16]),
        
        # 🔥 lr 改为离散选择（对数空间）
        'lr': trial.suggest_categorical('lr', [1e-5, 1e-4]),
        
        # 🔥 weight_decay 改为离散选择
        'weight_decay': trial.suggest_categorical('weight_decay', [1e-6, 1e-5, 1e-4]),
        
        'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw']),
        
        # 🔥 ranking_weight 改为离散选择（精简）
        'ranking_weight': trial.suggest_categorical('ranking_weight', [0.0, 0.1, 0.2, 0.3]),
        
        # 🔥 gc 精简到关键值（梯度累积步数一般不需要太多选择）
        'gc': trial.suggest_categorical('gc', [8, 16, 32]),
    })
    
    print(f"\n{'#'*60}")
    print(f"# Trial {trial.number} 参数:")
    print(f"{'#'*60}")
    for key, value in params.items():
        if key in trial.params:
            print(f"  {key}: {value}")
    print(f"{'#'*60}\n")
    
    val_cindex = run_ddp_training(params, trial.number)
    
    # 🔥 返回 None 就跳过
    if val_cindex is None:
        raise optuna.TrialPruned()
    
    return val_cindex


def objective_stage2(trial, best_stage1_params):
    """阶段2: 损失函数+调度器"""
    params = FIXED_PARAMS.copy()
    params.update(best_stage1_params)
    
    scheduler = trial.suggest_categorical('scheduler', ['cosine', 'step', 'plateau'])
    params['scheduler'] = scheduler
    
    if scheduler == 'cosine':
        params['min_lr'] = trial.suggest_float('min_lr', 1e-7, 1e-5, log=True)
    elif scheduler == 'step':
        params['step_size'] = trial.suggest_int('step_size', 20, 50)
        params['gamma'] = trial.suggest_float('gamma', 0.1, 0.5)
    
    params['ranking_weight'] = trial.suggest_float('ranking_weight', 0.0, 0.5)
    
    print(f"\n{'#'*60}")
    print(f"# Trial {trial.number} 参数 (阶段2):")
    print(f"{'#'*60}")
    for key, value in params.items():
        if key in trial.params:
            print(f"  {key}: {value}")
    print(f"{'#'*60}\n")
    
    val_cindex = run_ddp_training(params, trial.number)
    
    if val_cindex is None:
        raise optuna.TrialPruned()
    
    return val_cindex


def objective_stage3(trial, best_stage2_params):
    """阶段3: 正则化微调"""
    params = FIXED_PARAMS.copy()
    params.update(best_stage2_params)
    
    # 🔥 修复 dropout 范围问题
    dropout_center = best_stage2_params['dropout']
    params['dropout'] = trial.suggest_float('dropout', 
        max(0.3, dropout_center - 0.1),
        min(0.9, dropout_center + 0.1)
    )
    
    params['weight_decay'] = trial.suggest_float('weight_decay',
        best_stage2_params['weight_decay'] * 0.5,
        best_stage2_params['weight_decay'] * 2.0,
        log=True
    )
    
    params['gc'] = trial.suggest_int('gc',
        max(1, best_stage2_params['gc'] - 8),
        min(32, best_stage2_params['gc'] + 8)
    )
    
    print(f"\n{'#'*60}")
    print(f"# Trial {trial.number} 参数 (阶段3):")
    print(f"{'#'*60}")
    for key, value in params.items():
        if key in trial.params:
            print(f"  {key}: {value}")
    print(f"{'#'*60}\n")
    
    val_cindex = run_ddp_training(params, trial.number)
    
    if val_cindex is None:
        raise optuna.TrialPruned()
    
    return val_cindex


# ===================== 主函数 =====================
def main():
    """三阶段优化主流程"""
    
    setup_environment()
    
    start_time = time.time()
    
    os.makedirs("./results/optuna_study", exist_ok=True)
    
    storage = f"sqlite:///{STORAGE_PATH}"
    
    print("\n" + "="*60)
    print("🎯 阶段1: 核心架构优化")
    print("="*60)
    
    study1 = optuna.create_study(
        study_name=f"{STUDY_NAME}_stage1",
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    study1.optimize(objective_stage1, n_trials=30, n_jobs=N_JOBS)
    
    best_stage1 = study1.best_trial
    print(f"\n{'='*60}")
    print(f"✅ 阶段1最佳结果:")
    print(f"{'='*60}")
    print(f"   C-Index: {best_stage1.value:.4f}")
    print(f"   参数:")
    for key, value in best_stage1.params.items():
        print(f"     {key}: {value}")
    print(f"{'='*60}\n")
    
    with open("./results/optuna_study/stage1_best.json", 'w') as f:
        json.dump({
            'value': best_stage1.value,
            'params': best_stage1.params
        }, f, indent=2)
    
    print("\n" + "="*60)
    print("🎯 阶段2: 损失函数+调度器优化")
    print("="*60)
    
    study2 = optuna.create_study(
        study_name=f"{STUDY_NAME}_stage2",
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=5)
    )
    
    study2.optimize(
        lambda trial: objective_stage2(trial, best_stage1.params),
        n_trials=20,
        n_jobs=N_JOBS
    )
    
    best_stage2 = study2.best_trial
    print(f"\n{'='*60}")
    print(f"✅ 阶段2最佳结果:")
    print(f"{'='*60}")
    print(f"   C-Index: {best_stage2.value:.4f}")
    print(f"   参数:")
    for key, value in best_stage2.params.items():
        print(f"     {key}: {value}")
    print(f"{'='*60}\n")
    
    final_params = FIXED_PARAMS.copy()
    final_params.update(best_stage1.params)
    final_params.update(best_stage2.params)
    
    with open("./results/optuna_study/stage2_best.json", 'w') as f:
        json.dump({
            'value': best_stage2.value,
            'params': best_stage2.params,
            'full_params': final_params
        }, f, indent=2)
    
    print("\n" + "="*60)
    print("🎯 阶段3: 正则化微调")
    print("="*60)
    
    study3 = optuna.create_study(
        study_name=f"{STUDY_NAME}_stage3",
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=5)
    )
    
    study3.optimize(
        lambda trial: objective_stage3(trial, final_params),
        n_trials=15,
        n_jobs=N_JOBS
    )
    
    best_stage3 = study3.best_trial
    print(f"\n{'='*60}")
    print(f"✅ 阶段3最佳结果:")
    print(f"{'='*60}")
    print(f"   C-Index: {best_stage3.value:.4f}")
    print(f"   参数:")
    for key, value in best_stage3.params.items():
        print(f"     {key}: {value}")
    print(f"{'='*60}\n")
    
    final_best_params = FIXED_PARAMS.copy()
    final_best_params.update(best_stage1.params)
    final_best_params.update(best_stage2.params)
    final_best_params.update(best_stage3.params)
    
    with open("./results/optuna_study/final_best.json", 'w') as f:
        json.dump({
            'value': best_stage3.value,
            'params': best_stage3.params,
            'full_params': final_best_params
        }, f, indent=2)
    
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    
    print("\n" + "="*60)
    print("🎉 三阶段优化完成!")
    print("="*60)
    print(f"总耗时: {hours}h {minutes}m {seconds}s")
    print(f"\n最终最佳参数:")
    print(json.dumps(final_best_params, indent=2))
    print(f"\n最佳 C-Index: {best_stage3.value:.4f}")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断，正在清理资源...")
        cleanup_resources()
        print("✓ 清理完成")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        cleanup_resources()
        sys.exit(1)
