#!/bin/bash

# ============================================================
# Optuna 超参数优化启动脚本 (DDP版本)
# ============================================================
# ===================== MKL线程库修复 =====================
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1
# 设置缓存路径 (借鉴DDP脚本)

export HOME_CACHE="/home/stat-jijianxin/.cache"
mkdir -p $HOME_CACHE

export TRITON_CACHE_DIR="$HOME_CACHE/triton"
export TORCH_COMPILE_CACHE_DIR="$HOME_CACHE/torch_compile"
export TRANSFORMERS_CACHE="$HOME_CACHE/transformers"
export HF_HOME="$HOME_CACHE/huggingface"

mkdir -p $TRITON_CACHE_DIR
mkdir -p $TORCH_COMPILE_CACHE_DIR

# 清理旧缓存
rm -rf $TRITON_CACHE_DIR/* 2>/dev/null || true

echo "✓ 缓存路径已设置到: $HOME_CACHE"

# NCCL配置 (借鉴DDP脚本)
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800

# GPU配置
export CUDA_VISIBLE_DEVICES=0,1

# 环境检查
echo ""
echo "============================================================"
echo "🔍 环境检查"
echo "============================================================"
python3 --version
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'GPU 数量: {torch.cuda.device_count()}')"
python3 -c "import optuna; print(f'Optuna: {optuna.__version__}')"

# 检查磁盘空间
echo ""
echo "磁盘空间检查:"
df -h /home/stat-jijianxin | tail -1
df -h /tmp | tail -1

echo ""
echo "✓ 环境检查通过"
echo ""

# 启动优化
echo "============================================================"
echo "🚀 启动 Optuna 超参数优化 (DDP版本)"
echo "============================================================"
echo ""

START_TIME=$(date +%s)

python3 optuna_optimize.py 2>&1 | tee ./results/optuna_study/optimization.log

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "============================================================"
echo "✅ 优化完成!"
echo "============================================================"
echo "总耗时: $((ELAPSED/3600))h $((ELAPSED%3600/60))m $((ELAPSED%60))s"
echo "结果保存至: ./results/optuna_study/"
echo "============================================================"
