#!/bin/bash

# ============================================================
# 生存分析训练脚本 - DDP修复版
# 修复: DDP多GPU训练的参数传递问题
# ============================================================

set -e
set -u
set -o pipefail

# ============================================================
# 全局配置
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_ROOT}/logs/${TIMESTAMP}"
mkdir -p "$LOG_DIR"

MAIN_LOG="${LOG_DIR}/main.log"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ============================================================
# 日志函数
# ============================================================
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        INFO)
            echo -e "${BLUE}[INFO]${NC} ${timestamp} - ${message}" | tee -a "$MAIN_LOG"
            ;;
        SUCCESS)
            echo -e "${GREEN}[SUCCESS]${NC} ${timestamp} - ${message}" | tee -a "$MAIN_LOG"
            ;;
        WARNING)
            echo -e "${YELLOW}[WARNING]${NC} ${timestamp} - ${message}" | tee -a "$MAIN_LOG"
            ;;
        ERROR)
            echo -e "${RED}[ERROR]${NC} ${timestamp} - ${message}" | tee -a "$MAIN_LOG"
            ;;
    esac
}

# ============================================================
# 环境设置
# ============================================================
setup_environment() {
    log INFO "设置环境变量..."
    
    export HOME_CACHE="${HOME}/.cache/survival_training"
    mkdir -p "$HOME_CACHE"
    
    export TRITON_CACHE_DIR="${HOME_CACHE}/triton"
    export TORCH_COMPILE_CACHE_DIR="${HOME_CACHE}/torch_compile"
    export TRANSFORMERS_CACHE="${HOME_CACHE}/transformers"
    export HF_HOME="${HOME_CACHE}/huggingface"
    export TORCH_HOME="${HOME_CACHE}/torch"
    
    mkdir -p "$TRITON_CACHE_DIR" "$TORCH_COMPILE_CACHE_DIR"
    
    export NCCL_SOCKET_IFNAME=lo
    export NCCL_IB_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_SHM_DISABLE=0
    export NCCL_BLOCKING_WAIT=1
    export NCCL_ASYNC_ERROR_HANDLING=1
    export NCCL_DEBUG=WARN
    export NCCL_TIMEOUT=1800
    
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    
    log SUCCESS "环境变量设置完成"
}

# ============================================================
# 资源清理
# ============================================================
cleanup_resources() {
    local level=${1:-"normal"}
    
    log INFO "清理资源 (级别: $level)..."
    
    pkill -9 -f "torchrun" 2>/dev/null || true
    pkill -9 -f "main2.py" 2>/dev/null || true
    sleep 2
    
    python3 << 'PYEOF' 2>/dev/null || true
import torch
import gc
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
gc.collect()
PYEOF
    
    rm -rf /dev/shm/torch_* 2>/dev/null || true
    
    if [ "$level" == "deep" ]; then
        rm -rf "$TRITON_CACHE_DIR"/* 2>/dev/null || true
        rm -rf /tmp/triton_cache_rank_* 2>/dev/null || true
        rm -rf /tmp/torch_* 2>/dev/null || true
    fi
    
    sleep 2
    log SUCCESS "资源清理完成"
}

# ============================================================
# 端口检查
# ============================================================
check_port_available() {
    local port=$1
    local max_wait=${2:-30}
    local waited=0
    
    while netstat -tuln 2>/dev/null | grep -q ":$port "; do
        if [ $waited -ge $max_wait ]; then
            log WARNING "端口 $port 仍被占用,强制释放..."
            fuser -k $port/tcp 2>/dev/null || true
            sleep 2
            return 1
        fi
        log INFO "等待端口 $port 释放... ($waited/$max_wait)"
        sleep 1
        waited=$((waited + 1))
    done
    
    return 0
}

# ============================================================
# 环境检查
# ============================================================
check_environment() {
    log INFO "检查运行环境..."
    
    if ! command -v python3 &> /dev/null; then
        log ERROR "Python3 未安装"
        exit 1
    fi
    
    local python_version=$(python3 --version)
    log INFO "Python 版本: $python_version"
    
    if ! python3 -c "import torch" 2>/dev/null; then
        log ERROR "PyTorch 未安装"
        exit 1
    fi
    
    local torch_version=$(python3 -c "import torch; print(torch.__version__)")
    log INFO "PyTorch 版本: $torch_version"
    
    local cuda_available=$(python3 -c "import torch; print(torch.cuda.is_available())")
    local gpu_count=$(python3 -c "import torch; print(torch.cuda.device_count())")
    
    if [ "$cuda_available" == "True" ]; then
        log SUCCESS "CUDA 可用, GPU 数量: $gpu_count"
        
        for i in $(seq 0 $((gpu_count-1))); do
            local gpu_name=$(python3 -c "import torch; print(torch.cuda.get_device_name($i))")
            log INFO "  GPU $i: $gpu_name"
        done
    else
        log WARNING "CUDA 不可用,将使用 CPU 训练"
    fi
    
    if [ ! -f "${PROJECT_ROOT}/main2.py" ]; then
        log ERROR "main2.py 不存在: ${PROJECT_ROOT}/main2.py"
        exit 1
    fi
    log SUCCESS "找到训练脚本: ${PROJECT_ROOT}/main2.py"
    
    log SUCCESS "环境检查通过"
}

# ============================================================
# 数据检查
# ============================================================
check_data() {
    local csv_path=$1
    local h5_dir=$2
    
    log INFO "检查数据文件..."
    
    if [ ! -f "$csv_path" ]; then
        log ERROR "CSV 文件不存在: $csv_path"
        exit 1
    fi
    
    if [ ! -d "$h5_dir" ]; then
        log ERROR "H5 目录不存在: $h5_dir"
        exit 1
    fi
    
    local sample_count=$(tail -n +2 "$csv_path" | wc -l)
    log INFO "样本数量: $sample_count"
    
    local h5_count=$(find "$h5_dir" -name "*.h5" 2>/dev/null | wc -l)
    log INFO "H5 文件总数: $h5_count"
    
    local feature_dir="${h5_dir}/features_ctranspath"
    if [ -d "$feature_dir" ]; then
        local feature_h5_count=$(find "$feature_dir" -name "*.h5" 2>/dev/null | wc -l)
        log SUCCESS "features_ctranspath: $feature_h5_count 个文件"
    else
        log WARNING "features_ctranspath 目录不存在: $feature_dir"
    fi
    
    log SUCCESS "数据检查通过"
}

# ============================================================
# 训练单个 Fold - 🔥 关键修复
# ============================================================
train_fold() {
    local fold=$1
    local config=$2
    
    log INFO "开始训练 Fold $fold"
    
    # 🔥 解析所有配置到变量 (避免引号嵌套问题)
    local csv_path=$(echo "$config" | jq -r '.csv_path')
    local h5_base_dir=$(echo "$config" | jq -r '.h5_base_dir')
    local external_csv=$(echo "$config" | jq -r '.external_csv_path // empty')
    local external_h5=$(echo "$config" | jq -r '.external_h5_base_dir // empty')
    local feature_models=$(echo "$config" | jq -r '.feature_models | join(" ")')
    local label_col=$(echo "$config" | jq -r '.label_col // "disc_label"')
    
    local model_version=$(echo "$config" | jq -r '.model_version // "standard"')
    local in_dim=$(echo "$config" | jq -r '.in_dim')
    local n_classes=$(echo "$config" | jq -r '.n_classes')
    local dropout=$(echo "$config" | jq -r '.dropout')
    local drop_path_rate=$(echo "$config" | jq -r '.drop_path_rate // 0.1')
    local feature_dropout=$(echo "$config" | jq -r '.feature_dropout // 0.1')
    local act=$(echo "$config" | jq -r '.act // "gelu"')
    local mamba_layer=$(echo "$config" | jq -r '.mamba_layer')
    
    local batch_size=$(echo "$config" | jq -r '.batch_size')
    local max_epochs=$(echo "$config" | jq -r '.max_epochs')
    local lr=$(echo "$config" | jq -r '.lr')
    local weight_decay=$(echo "$config" | jq -r '.weight_decay')
    local optimizer=$(echo "$config" | jq -r '.optimizer')
    local loss=$(echo "$config" | jq -r '.loss')
    local alpha_surv=$(echo "$config" | jq -r '.alpha_surv // 0.0')
    local gc=$(echo "$config" | jq -r '.gc // 1')
    
    local max_grad_norm=$(echo "$config" | jq -r '.max_grad_norm // 1.0')
    local feature_drop_rate=$(echo "$config" | jq -r '.feature_drop_rate // 0.1')
    local label_smoothing=$(echo "$config" | jq -r '.label_smoothing // 0.0')
    
    local scheduler=$(echo "$config" | jq -r '.scheduler // "cosine"')
    local lr_step_size=$(echo "$config" | jq -r '.lr_step_size // 30')
    local lr_gamma=$(echo "$config" | jq -r '.lr_gamma // 0.5')
    local warmup_epochs=$(echo "$config" | jq -r '.warmup_epochs // 0')
    
    local ranking_weight=$(echo "$config" | jq -r '.ranking_weight // 0.1')
    local ranking_margin=$(echo "$config" | jq -r '.ranking_margin // 0.0')
    
    local warmup=$(echo "$config" | jq -r '.warmup // 5')
    local patience=$(echo "$config" | jq -r '.patience // 15')
    local stop_epoch=$(echo "$config" | jq -r '.stop_epoch // 20')
    local early_stop_delta=$(echo "$config" | jq -r '.early_stop_delta // 0.0001')
    local save_all=$(echo "$config" | jq -r '.save_all_checkpoints // false')
    
    local k_fold=$(echo "$config" | jq -r '.k_fold')
    local val_ratio=$(echo "$config" | jq -r '.val_ratio // 0.15')
    local test_ratio=$(echo "$config" | jq -r '.test_ratio // 0.15')
    
    local results_dir=$(echo "$config" | jq -r '.results_dir')
    local num_workers=$(echo "$config" | jq -r '.num_workers // 0')
    local seed=$(echo "$config" | jq -r '.seed // 42')
    local num_gpus=$(echo "$config" | jq -r '.num_gpus // 1')
    
    # 动态端口
    local master_port=$((29500 + fold))
    
    # 检查端口
    if ! check_port_available $master_port 30; then
        log ERROR "端口 $master_port 不可用"
        return 1
    fi
    
    # 清理资源
    cleanup_resources "normal"
    
    # Fold 日志文件
    local fold_log="${LOG_DIR}/fold_${fold}.log"
    
    # 记录开始时间
    local start_time=$(date +%s)
    
    # 🔥 构建命令数组 (避免引号问题)
    local cmd_args=(
        "${PROJECT_ROOT}/main2.py"
        --csv_path "$csv_path"
        --h5_base_dir "$h5_base_dir"
        --feature_models $feature_models
        --label_col "$label_col"
        --model_version "$model_version"
        --in_dim "$in_dim"
        --n_classes "$n_classes"
        --dropout "$dropout"
        --drop_path_rate "$drop_path_rate"
        --feature_dropout "$feature_dropout"
        --act "$act"
        --mamba_layer "$mamba_layer"
        --batch_size "$batch_size"
        --max_epochs "$max_epochs"
        --lr "$lr"
        --weight_decay "$weight_decay"
        --optimizer "$optimizer"
        --loss "$loss"
        --alpha_surv "$alpha_surv"
        --gc "$gc"
        --max_grad_norm "$max_grad_norm"
        --feature_drop_rate "$feature_drop_rate"
        --label_smoothing "$label_smoothing"
        --scheduler "$scheduler"
        --lr_step_size "$lr_step_size"
        --lr_gamma "$lr_gamma"
        --warmup_epochs "$warmup_epochs"
        --ranking_weight "$ranking_weight"
        --ranking_margin "$ranking_margin"
        --warmup "$warmup"
        --patience "$patience"
        --stop_epoch "$stop_epoch"
        --early_stop_delta "$early_stop_delta"
        --k_fold "$k_fold"
        --fold "$fold"
        --val_ratio "$val_ratio"
        --test_ratio "$test_ratio"
        --results_dir "$results_dir"
        --num_workers "$num_workers"
        --seed "$seed"
    )
    
    # 可选参数
    if [ -n "$external_csv" ] && [ -n "$external_h5" ]; then
        cmd_args+=(--external_csv_path "$external_csv")
        cmd_args+=(--external_h5_base_dir "$external_h5")
    fi
    
    if [ "$save_all" == "true" ]; then
        cmd_args+=(--save_all_checkpoints)
    fi
    
    # GPU 配置
    if [ $num_gpus -gt 1 ]; then
        local gpu_ids=$(seq -s, 0 $((num_gpus-1)))
        export CUDA_VISIBLE_DEVICES="$gpu_ids"
        log INFO "使用 GPU: $gpu_ids (DDP模式)"
    else
        export CUDA_VISIBLE_DEVICES="0"
        log INFO "使用 GPU: 0 (单卡模式)"
    fi
    
    log INFO "启动训练 (端口: $master_port, GPU: $num_gpus)"
    log INFO "日志文件: $fold_log"
    
    # 🔥 启动训练 (使用数组避免引号问题)
    if [ $num_gpus -gt 1 ]; then
        # DDP 训练
        timeout 7200 torchrun \
            --nproc_per_node=$num_gpus \
            --master_addr=127.0.0.1 \
            --master_port=$master_port \
            --node_rank=0 \
            --nnodes=1 \
            "${cmd_args[@]}" > "$fold_log" 2>&1
    else
        # 单 GPU 训练
        timeout 7200 python3 "${cmd_args[@]}" > "$fold_log" 2>&1
    fi
    
    local exit_code=$?
    
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    
    if [ $exit_code -eq 0 ]; then
        log SUCCESS "Fold $fold 训练成功 (耗时: $((elapsed/60))m $((elapsed%60))s)"
        return 0
    elif [ $exit_code -eq 124 ]; then
        log ERROR "Fold $fold 训练超时 (2小时)"
        log ERROR "查看日志: $fold_log"
        return 1
    else
        log ERROR "Fold $fold 训练失败 (退出码: $exit_code)"
        log ERROR "查看日志: $fold_log"
        log ERROR "最后20行日志:"
        tail -20 "$fold_log" 2>/dev/null | while read line; do
            log ERROR "  $line"
        done
        return 1
    fi
}

# ============================================================
# K-Fold 训练
# ============================================================
train_kfold() {
    local config=$1
    
    local k_fold=$(echo "$config" | jq -r '.k_fold')
    local results_dir=$(echo "$config" | jq -r '.results_dir')
    
    log INFO "开始 K-Fold 训练 (K=$k_fold)"
    
    mkdir -p "$results_dir"
    echo "$config" | jq '.' > "${results_dir}/config.json"
    
    local failed_folds=()
    local start_time=$(date +%s)
    
    for fold in $(seq 0 $((k_fold-1))); do
        log INFO "========== Fold $fold / $((k_fold-1)) =========="
        
        if train_fold $fold "$config"; then
            log SUCCESS "Fold $fold 完成"
        else
            log ERROR "Fold $fold 失败"
            failed_folds+=($fold)
        fi
        
        cleanup_resources "normal"
        
        if [ $fold -lt $((k_fold-1)) ]; then
            log INFO "等待 10 秒后开始下一个 Fold..."
            sleep 10
        fi
    done
    
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    
    # 汇总结果
    log INFO "汇总 K-Fold 结果..."
    
    python3 << EOF
import os, pickle, pandas as pd, numpy as np, json

results_dir = "$results_dir"
k_fold = $k_fold

all_results = []
for fold in range(k_fold):
    results_file = os.path.join(results_dir, f'fold_{fold}', 'results.pkl')
    if os.path.exists(results_file):
        try:
            with open(results_file, 'rb') as f:
                all_results.append(pickle.load(f))
        except Exception as e:
            print(f"⚠️  Fold {fold} 结果文件损坏: {e}")

if all_results:
    val_ci = [r['val_cindex'] for r in all_results]
    test_ci = [r['test_cindex'] for r in all_results]
    
    df = pd.DataFrame({
        'fold': list(range(len(all_results))),
        'val_cindex': val_ci,
        'test_cindex': test_ci
    })
    
    if 'external_cindex' in all_results[0]:
        df['external_cindex'] = [r['external_cindex'] for r in all_results]
    
    df.to_csv(os.path.join(results_dir, 'summary.csv'), index=False)
    
    summary = {
        'completed_folds': len(all_results),
        'total_folds': k_fold,
        'val_cindex_mean': float(np.mean(val_ci)),
        'val_cindex_std': float(np.std(val_ci)),
        'test_cindex_mean': float(np.mean(test_ci)),
        'test_cindex_std': float(np.std(test_ci))
    }
    
    with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f'\n✓ 完成 {len(all_results)}/{k_fold} Folds')
    print(f'验证集 C-Index: {np.mean(val_ci):.4f} ± {np.std(val_ci):.4f}')
    print(f'测试集 C-Index: {np.mean(test_ci):.4f} ± {np.std(test_ci):.4f}')
    print('\n' + df.to_string(index=False))
else:
    print('\n❌ 没有找到任何完成的 Fold 结果')
    exit(1)
EOF
    
    if [ ${#failed_folds[@]} -eq 0 ]; then
        log SUCCESS "所有 Fold 训练成功"
    else
        log WARNING "部分 Fold 训练失败: ${failed_folds[@]}"
    fi
    
    log INFO "总耗时: $((elapsed/3600))h $((elapsed%3600/60))m $((elapsed%60))s"
    log INFO "结果保存至: $results_dir"
    
    return ${#failed_folds[@]}
}

# ============================================================
# 参数搜索
# ============================================================
run_parameter_search() {
    local base_config=$1
    local param_grid=$2
    
    log INFO "开始参数搜索..."
    
    local batch_sizes=($(echo "$param_grid" | jq -r '.batch_sizes[]'))
    local learning_rates=($(echo "$param_grid" | jq -r '.learning_rates[]'))
    
    local total_configs=$((${#batch_sizes[@]} * ${#learning_rates[@]}))
    local current_config=0
    
    log INFO "参数组合数量: $total_configs"
    
    local global_start=$(date +%s)
    
    for batch_size in "${batch_sizes[@]}"; do
        for lr in "${learning_rates[@]}"; do
            current_config=$((current_config + 1))
            
            log INFO "========== 参数组合 $current_config / $total_configs =========="
            log INFO "Batch Size: $batch_size, Learning Rate: $lr"
            
            local exp_name="batch${batch_size}_lr${lr}"
            local results_dir="${PROJECT_ROOT}/results/${exp_name}_${TIMESTAMP}"
            
            local config=$(echo "$base_config" | jq \
                --arg bs "$batch_size" \
                --arg lr "$lr" \
                --arg rd "$results_dir" \
                '.batch_size = ($bs | tonumber) | .lr = ($lr | tonumber) | .results_dir = $rd')
            
            if train_kfold "$config"; then
                log SUCCESS "参数组合 $current_config 完成"
            else
                log ERROR "参数组合 $current_config 失败"
            fi
            
            cleanup_resources "deep"
            
            if [ $current_config -lt $total_configs ]; then
                log INFO "等待 30 秒后开始下一个参数组合..."
                sleep 30
            fi
        done
    done
    
    local global_end=$(date +%s)
    local global_elapsed=$((global_end - global_start))
    
    log SUCCESS "参数搜索完成"
    log INFO "总耗时: $((global_elapsed/3600))h $((global_elapsed%3600/60))m $((global_elapsed%60))s"
}

# ============================================================
# 主函数
# ============================================================
main() {
    log INFO "=========================================="
    log INFO "生存分析训练脚本 (DDP修复版)"
    log INFO "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    log INFO "=========================================="
    
    setup_environment
    check_environment
    
    # 基础配置
    local base_config=$(cat << 'EOF'
{
  "csv_path": "/home/stat-jijianxin/PFMs/Survival_code/csv_file/hmu_survival_with_slides.csv",
  "h5_base_dir": "/home/stat-jijianxin/PFMs/HMU_GC_ALL_H5",
  "external_csv_path": "/home/stat-jijianxin/PFMs/Survival_code/csv_file/tcga_survival_matched.csv",
  "external_h5_base_dir": "/home/stat-jijianxin/PFMs/TRIDENT/tcga_filtered/20x_512px_0px_overlap",
  "feature_models": ["ctranspath"],
  "label_col": "disc_label",
  "model_version": "standard",
  "in_dim": 768,
  "n_classes": 4,
  "dropout": 0.4,
  "drop_path_rate": 0.1,
  "feature_dropout": 0.1,
  "act": "gelu",
  "mamba_layer": 2,
  "max_epochs": 100,
  "weight_decay": 0.001,
  "optimizer": "adamw",
  "loss": "combined",
  "alpha_surv": 0.35,
  "ranking_weight": 0.1,
  "ranking_margin": 0.0,
  "gc": 1,
  "max_grad_norm": 1.0,
  "feature_drop_rate": 0.1,
  "label_smoothing": 0.0,
  "scheduler": "cosine",
  "lr_step_size": 30,
  "lr_gamma": 0.5,
  "warmup_epochs": 0,
  "k_fold": 10,
  "val_ratio": 0.1,
  "test_ratio": 0.1,
  "warmup": 5,
  "patience": 15,
  "stop_epoch": 20,
  "early_stop_delta": 0.0001,
  "save_all_checkpoints": false,
  "num_workers": 0,
  "seed": 42,
  "num_gpus": 2
}
EOF
)
    
    # 参数网格
    local param_grid=$(cat << 'EOF'
{
  "batch_sizes": [4, 8, 16],
  "learning_rates": [0.0001, 0.0002, 0.0005]
}
EOF
)
    
    local csv_path=$(echo "$base_config" | jq -r '.csv_path')
    local h5_dir=$(echo "$base_config" | jq -r '.h5_base_dir')
    check_data "$csv_path" "$h5_dir"
    
    run_parameter_search "$base_config" "$param_grid"
    
    log SUCCESS "=========================================="
    log SUCCESS "所有训练完成!"
    log SUCCESS "日志目录: $LOG_DIR"
    log SUCCESS "=========================================="
}

# ============================================================
# 信号处理
# ============================================================
trap 'log ERROR "脚本被中断"; cleanup_resources "deep"; exit 130' INT TERM

# ============================================================
# 执行主函数
# ============================================================
main "$@"
