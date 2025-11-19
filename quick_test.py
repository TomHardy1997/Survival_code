# quick_test.py (终极简化版 - 绝对可靠)
"""
返璞归真测试脚本
- 移除了所有复杂的、可能导致死锁的实时日志读取逻辑。
- 使用最稳定可靠的 subprocess 调用方式。
- 保留了所有正确的环境变量和清理逻辑。
"""

import os
import sys
import subprocess
import time

# ===================== 配置 (保持不变) =====================
NUM_GPUS = 2
CUDA_VISIBLE_DEVICES = "0,1"
MASTER_ADDR = "127.0.0.1"
MASTER_PORT = 29600
CSV_PATH = "/home/stat-jijianxin/PFMs/Survival_code/csv_file/hmu_survival_with_slides.csv"
H5_DIR = "/home/stat-jijianxin/PFMs/HMU_GC_ALL_H5"
EXTERNAL_CSV = "/home/stat-jijianxin/PFMs/Survival_code/csv_file/tcga_survival_matched.csv"
EXTERNAL_H5 = "/home/stat-jijianxin/PFMs/TRIDENT/tcga_filtered/20x_512px_0px_overlap"
TEST_PARAMS = {
    'dropout': 0.25, 'act': 'relu', 'mamba_layer': 2, 'batch_size': 4,
    'lr': 2e-4, 'weight_decay': 1e-5, 'optimizer': 'adamw',
    'ranking_weight': 0.1, 'gc': 1,
}

def cleanup_resources(deep=False):
    """清理资源 (工业级强化版)"""
    print("🧹 清理资源..." + (" (深度)" if deep else ""))
    subprocess.run("pkill -9 -f 'torchrun' 2>/dev/null || true", shell=True)
    subprocess.run("pkill -9 -f 'train_ddp.py' 2>/dev/null || true", shell=True)
    subprocess.run(f"fuser -k -n tcp {MASTER_PORT} 2>/dev/null || true", shell=True)
    
    print("  🧹 清理共享内存 (/dev/shm)...")
    subprocess.run("rm -rf /dev/shm/torch_* 2>/dev/null || true", shell=True)
    
    if deep:
        try:
            home_cache = os.path.expanduser("~/.cache")
            triton_cache = os.path.join(home_cache, "triton")
            if os.path.exists(triton_cache):
                print(f"  🧹 清理Triton缓存 ({triton_cache})...")
                subprocess.run(f"rm -rf {triton_cache}/* 2>/dev/null || true", shell=True)
        except Exception:
            pass
    
    time.sleep(2)
    print("✓ 清理完成")

def test_train_ddp():
    """测试 train_ddp.py - 使用最可靠的启动方式"""
    print("\n" + "="*60)
    print("🧪 测试 train_ddp.py (终极简化版)")
    print("="*60)
    
    cleanup_resources(deep=True)
    
    results_dir = "./test_results_1epoch"
    os.makedirs(results_dir, exist_ok=True)
    log_file = os.path.join(results_dir, "test.log")
    err_file = os.path.join(results_dir, "test.err")
    
    cmd = [
        "torchrun", f"--nproc_per_node={NUM_GPUS}", f"--master_addr={MASTER_ADDR}",
        f"--master_port={MASTER_PORT}", "train_ddp.py",
        "--csv_path", CSV_PATH, "--h5_dir", H5_DIR, "--external_csv_path", EXTERNAL_CSV,
        "--external_h5_dir", EXTERNAL_H5, "--feature_models", "ctranspath",
        "--batch_size", str(TEST_PARAMS['batch_size']), "--max_epochs", "1",
        "--lr", str(TEST_PARAMS['lr']), "--results_dir", results_dir,
        "--num_workers", "0", "--seed", "42",
        # ... 其他参数保持不变 ...
        "--in_dim", "768", "--n_classes", "4", "--dropout", str(TEST_PARAMS['dropout']),
        "--act", TEST_PARAMS['act'], "--mamba_layer", str(TEST_PARAMS['mamba_layer']),
        "--weight_decay", str(TEST_PARAMS['weight_decay']), "--optimizer", TEST_PARAMS['optimizer'],
        "--gc", str(TEST_PARAMS['gc']), "--loss", "combined", "--main_loss_type", "nll",
        "--alpha_surv", "0.365", "--ranking_weight", str(TEST_PARAMS['ranking_weight']),
        "--ranking_margin", "0.0", "--k_fold", "3", "--fold", "0",
        "--val_ratio", "0.2", "--test_ratio", "0.2", "--warmup", "0",
        "--patience", "999", "--stop_epoch", "1",
    ]
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
    env['NCCL_BLOCKING_WAIT'] = '1'
    env['NCCL_DEBUG'] = 'WARN'
    env['NCCL_TIMEOUT'] = '1800'
    env['NCCL_SHM_DISABLE'] = '0'
    env['NCCL_SOCKET_IFNAME'] = 'lo'
    env['NCCL_IB_DISABLE'] = '1'
    
    print(f"\n⚙️  环境配置:")
    print(f"  - NCCL_BLOCKING_WAIT: {env['NCCL_BLOCKING_WAIT']} (✅ 阻塞模式)")
    print(f"  - 日志将直接写入文件，控制台将保持安静。")
    print(f"\n📝 日志文件: {log_file}")
    print(f"📝 错误文件: {err_file}")
    print("\n🔥 开始执行... (超时 5 分钟)")
    print("\n💡 请打开新终端，使用以下命令实时查看日志:")
    print(f"   tail -f {log_file}")
    
    process = None
    start_time = time.time()
    try:
        # 🔥🔥🔥 核心修改：不再使用PIPE，直接重定向到文件 🔥🔥🔥
        with open(log_file, 'w') as f_out, open(err_file, 'w') as f_err:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=f_out,  # 直接写入文件
                stderr=f_err   # 直接写入文件
            )
            
            # 等待进程结束，设置5分钟超时
            returncode = process.wait(timeout=300)
        
        elapsed = time.time() - start_time
        
        if returncode == 0:
            print(f"\n✅ 训练成功! (耗时: {elapsed:.1f}秒)")
            print(f"请查看 {log_file} 获取详细输出。")
            return True
        else:
            print(f"\n❌ 训练失败! (Exit Code: {returncode}, 耗时: {elapsed:.1f}秒)")
            print("错误日志内容如下:")
            with open(err_file, 'r') as f:
                print(f.read() or "(错误日志为空)")
            return False

    except subprocess.TimeoutExpired:
        print("\n❌ 训练超时 (5分钟)")
        if process:
            process.kill()
        print("进程已被终止。请检查日志文件以确定卡在何处:")
        print(f"tail -100 {log_file}")
        print(f"tail -100 {err_file}")
        return False
    except Exception as e:
        print(f"\n❌ 启动训练时发生致命异常: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cleanup_resources()

if __name__ == "__main__":
    try:
        test_train_ddp()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        cleanup_resources(deep=True)
    finally:
        print("\n👋 测试结束")
