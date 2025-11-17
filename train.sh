#!/bin/bash

# ============================================================
# 生存分析训练脚本 - 完整强化版 (解决 /tmp 满问题)
# ============================================================

# ============================================================
# 🔥 关键: 修改所有缓存路径到 home 目录 (避免 /tmp 满)
# ============================================================
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

# ============================================================
# 🔥 NCCL 配置
# ============================================================
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800  # 30分钟超时

# ============================================================
# GPU 配置
# ============================================================
export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2
MASTER_ADDR=127.0.0.1
BASE_PORT=29500  # 动态端口基准

# ============================================================
# 数据路径
# ============================================================
CSV_PATH="/home/stat-jijianxin/PFMs/Survival_code/csv_file/hmu_survival_with_slides.csv"
H5_DIR="/home/stat-jijianxin/PFMs/HMU_GC_ALL_H5/features_ctranspath"
EXTERNAL_CSV="/home/stat-jijianxin/PFMs/Survival_code/csv_file/tcga_survival_matched.csv"
EXTERNAL_H5="/home/stat-jijianxin/PFMs/TRIDENT/tcga_filtered/20x_512px_0px_overlap/features_ctranspath"

# ============================================================
# 模型参数
# ============================================================
IN_DIM=768
N_CLASSES=4
DROPOUT=0.25
ACT="gelu"
MAMBA_LAYER=2

# ============================================================
# 训练参数
# ============================================================
MAX_EPOCHS=100
WEIGHT_DECAY=1e-5
OPTIMIZER="adamw"

# ============================================================
# 损失函数参数
# ============================================================
LOSS="combined"
ALPHA_SURV=0.35
RANKING_WEIGHT=0.1
RANKING_MARGIN=0.0
GC=1

# ============================================================
# K-Fold参数
# ============================================================
K_FOLD=10
VAL_RATIO=0.1
TEST_RATIO=0.1

# ============================================================
# 早停参数
# ============================================================
WARMUP=5
PATIENCE=15
STOP_EPOCH=20

# ============================================================
# 其他参数
# ============================================================
NUM_WORKERS=0
SEED=42

# ============================================================
# 🔥 参数组合
# ============================================================
PARAM_GROUPS=(
  "8 2e-4 results_hmu_tcga_ddp_batch8_lr2e4"
  "4 1e-4 results_hmu_tcga_ddp_batch4_lr1e4"
  "16 5e-4 results_hmu_tcga_ddp_batch16_lr5e4"
)

# ============================================================
# 🔥 强化清理函数
# ============================================================
cleanup_resources() {
    echo "🧹 强化资源清理..."
    
    # 1. 杀死所有相关进程
    pkill -9 -f "torchrun" 2>/dev/null || true
    pkill -9 -f "main.py" 2>/dev/null || true
    pkill -9 -f "python.*main.py" 2>/dev/null || true
    
    # 2. 清理 CUDA 缓存
    python3 << 'PYEOF'
import torch
import gc
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
gc.collect()
print("✓ CUDA 缓存已清理")
PYEOF
    
    # 3. 清理共享内存
    rm -rf /dev/shm/torch_* 2>/dev/null || true
    
    # 4. 🔥 清理 Triton 缓存 (关键!)
    rm -rf $TRITON_CACHE_DIR/* 2>/dev/null || true
    
    # 5. 🔥 清理 /tmp 中的临时文件
    rm -rf /tmp/triton_cache_rank_* 2>/dev/null || true
    rm -rf /tmp/torch_* 2>/dev/null || true
    
    # 6. 等待端口释放
    sleep 3
    
    echo "✓ 资源清理完成"
}

# ============================================================
# 🔥 检查端口函数
# ============================================================
wait_for_port() {
    local port=$1
    local max_wait=30
    local waited=0
    
    while netstat -tuln 2>/dev/null | grep -q ":$port "; do
        if [ $waited -ge $max_wait ]; then
            echo "⚠️  端口 $port 仍被占用，强制清理..."
            fuser -k $port/tcp 2>/dev/null || true
            sleep 2
            break
        fi
        echo "⏳ 等待端口 $port 释放... ($waited/$max_wait)"
        sleep 1
        waited=$((waited + 1))
    done
}

# ============================================================
# 环境检查
# ============================================================
echo ""
echo "============================================================"
echo "🔍 环境检查"
echo "============================================================"
python3 --version
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'GPU 数量: {torch.cuda.device_count()}')"
echo "缓存目录: $TRITON_CACHE_DIR"

if [ ! -f "$CSV_PATH" ]; then
    echo "❌ CSV文件不存在: $CSV_PATH"
    exit 1
fi
if [ ! -d "$H5_DIR" ]; then
    echo "❌ H5目录不存在: $H5_DIR"
    exit 1
fi

# 检查磁盘空间
echo ""
echo "磁盘空间检查:"
df -h /home/stat-jijianxin | tail -1
df -h /tmp | tail -1

echo ""
echo "✓ 环境检查通过"
echo ""

# ============================================================
# 🔥 主循环
# ============================================================
GLOBAL_START_TIME=$(date +%s)

for PARAM_GROUP in "${PARAM_GROUPS[@]}"; do
  IFS=' ' read -r BATCH_SIZE LR RESULTS_DIR <<< "$PARAM_GROUP"
  
  echo ""
  echo "============================================================"
  echo "🚀 开始训练参数组"
  echo "============================================================"
  echo "Batch Size: $BATCH_SIZE"
  echo "Learning Rate: $LR"
  echo "结果目录: $RESULTS_DIR"
  echo "============================================================"
  
  mkdir -p $RESULTS_DIR
  TOTAL_START_TIME=$(date +%s)
  FAILED_FOLDS=()

  # ============================================================
  # Fold 循环
  # ============================================================
  for FOLD in $(seq 0 $((K_FOLD-1))); do
      echo ""
      echo "============================================================"
      echo "📊 训练 Fold $FOLD / $((K_FOLD-1))"
      echo "============================================================"
      
      # 🔥 动态端口 (避免冲突)
      MASTER_PORT=$((BASE_PORT + FOLD))
      
      # 🔥 清理资源
      cleanup_resources
      
      # 🔥 等待端口
      wait_for_port $MASTER_PORT
      
      FOLD_START_TIME=$(date +%s)
      
      # 构建命令
      CMD="main.py \
          --csv_path $CSV_PATH \
          --h5_dir $H5_DIR \
          --in_dim $IN_DIM \
          --n_classes $N_CLASSES \
          --dropout $DROPOUT \
          --act $ACT \
          --mamba_layer $MAMBA_LAYER \
          --batch_size $BATCH_SIZE \
          --max_epochs $MAX_EPOCHS \
          --lr $LR \
          --weight_decay $WEIGHT_DECAY \
          --optimizer $OPTIMIZER \
          --loss $LOSS \
          --alpha_surv $ALPHA_SURV \
          --ranking_weight $RANKING_WEIGHT \
          --ranking_margin $RANKING_MARGIN \
          --gc $GC \
          --k_fold $K_FOLD \
          --fold $FOLD \
          --val_ratio $VAL_RATIO \
          --test_ratio $TEST_RATIO \
          --warmup $WARMUP \
          --patience $PATIENCE \
          --stop_epoch $STOP_EPOCH \
          --results_dir $RESULTS_DIR \
          --num_workers $NUM_WORKERS \
          --seed $SEED"
      
      if [ ! -z "$EXTERNAL_CSV" ] && [ ! -z "$EXTERNAL_H5" ]; then
          CMD="$CMD --external_csv_path $EXTERNAL_CSV --external_h5_dir $EXTERNAL_H5"
      fi
      
      echo "启动训练 (端口: $MASTER_PORT)..."
      
      # 🔥 启动训练 (带超时保护)
      timeout 7200 torchrun \
          --nproc_per_node=$NUM_GPUS \
          --master_addr=$MASTER_ADDR \
          --master_port=$MASTER_PORT \
          --node_rank=0 \
          --nnodes=1 \
          $CMD
      
      EXIT_CODE=$?
      
      FOLD_END_TIME=$(date +%s)
      FOLD_ELAPSED=$((FOLD_END_TIME - FOLD_START_TIME))
      
      echo ""
      if [ $EXIT_CODE -eq 0 ]; then
          echo "✅ Fold $FOLD 训练成功"
          echo "耗时: $((FOLD_ELAPSED/60)) 分钟 $((FOLD_ELAPSED%60)) 秒"
      elif [ $EXIT_CODE -eq 124 ]; then
          echo "⏱️  Fold $FOLD 训练超时 (2小时)"
          FAILED_FOLDS+=($FOLD)
      else
          echo "❌ Fold $FOLD 训练失败 (退出码: $EXIT_CODE)"
          FAILED_FOLDS+=($FOLD)
      fi
      
      # 🔥 Fold 间清理
      cleanup_resources
      echo "⏳ 等待 15 秒后开始下一个 Fold..."
      sleep 15
  done

  # ============================================================
  # 汇总结果
  # ============================================================
  echo ""
  echo "============================================================"
  echo "📊 汇总当前参数组 K-Fold 结果"
  echo "============================================================"
  
  export RESULTS_DIR
  export K_FOLD
  
  python3 << 'EOF'
import os
import pickle
import pandas as pd
import numpy as np

results_dir = os.environ['RESULTS_DIR']
k_fold = int(os.environ['K_FOLD'])

all_results = []
missing_folds = []

for fold in range(k_fold):
    results_file = os.path.join(results_dir, f'fold_{fold}', 'results.pkl')
    if os.path.exists(results_file):
        try:
            with open(results_file, 'rb') as f:
                all_results.append(pickle.load(f))
        except Exception as e:
            print(f"⚠️  Fold {fold} 结果文件损坏: {e}")
            missing_folds.append(fold)
    else:
        print(f"⚠️  Fold {fold} 结果文件不存在")
        missing_folds.append(fold)

if all_results:
    val_ci = [r['val_cindex'] for r in all_results]
    test_ci = [r['test_cindex'] for r in all_results]
    
    df = pd.DataFrame({
        'fold': list(range(len(all_results))),
        'val_cindex': val_ci,
        'test_cindex': test_ci
    })
    
    if 'external_cindex' in all_results[0]:
        ext_ci = [r['external_cindex'] for r in all_results]
        df['external_cindex'] = ext_ci
    
    summary_path = os.path.join(results_dir, 'summary.csv')
    df.to_csv(summary_path, index=False)
    
    print(f'\n✓ 完成 {len(all_results)}/{k_fold} Folds')
    print(f'\n验证集 C-index: {np.mean(val_ci):.4f} ± {np.std(val_ci):.4f}')
    print(f'测试集 C-index: {np.mean(test_ci):.4f} ± {np.std(test_ci):.4f}')
    
    if 'external_cindex' in all_results[0]:
        print(f'外部集 C-index: {np.mean(ext_ci):.4f} ± {np.std(ext_ci):.4f}')
    
    print('\n详细结果:')
    print(df.to_string(index=False))
    print(f'\n结果已保存至: {summary_path}')
else:
    print('\n❌ 没有找到任何完成的 Fold 结果')
EOF

  TOTAL_END_TIME=$(date +%s)
  TOTAL_ELAPSED=$((TOTAL_END_TIME - TOTAL_START_TIME))
  
  echo ""
  echo "============================================================"
  if [ ${#FAILED_FOLDS[@]} -eq 0 ]; then
      echo "✅ 当前参数组所有 Fold 训练成功"
  else
      echo "⚠️  当前参数组部分 Fold 训练失败: ${FAILED_FOLDS[@]}"
  fi
  echo "============================================================"
  echo "当前参数组总耗时: $((TOTAL_ELAPSED/3600))h $((TOTAL_ELAPSED%3600/60))m $((TOTAL_ELAPSED%60))s"
  echo "结果保存至: $RESULTS_DIR"
  echo ""
  
  # 🔥 参数组间强化清理
  cleanup_resources
  echo "⏳ 参数组切换，等待 30 秒，确保资源完全释放..."
  sleep 30
  echo "✓ 参数组间资源清理完成"
  echo ""
done

# ============================================================
# 最终统计
# ============================================================
GLOBAL_END_TIME=$(date +%s)
GLOBAL_ELAPSED=$((GLOBAL_END_TIME - GLOBAL_START_TIME))

echo ""
echo "============================================================"
echo "✅ 所有参数组训练完成!"
echo "============================================================"
echo "所有参数组总耗时: $((GLOBAL_ELAPSED/3600))h $((GLOBAL_ELAPSED%3600/60))m $((GLOBAL_ELAPSED%60))s"
echo "============================================================"
echo ""
