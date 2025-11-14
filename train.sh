#!/bin/bash

# ============================================================
# 生存分析训练脚本
# ============================================================

# ========== 数据路径 ==========
CSV_PATH="/home/stat-jijianxin/PFMs/Survival_code/csv_file/hmu_survival_with_slides.csv"
H5_DIR="/home/stat-jijianxin/PFMs/HMU_GC_ALL_H5/features_ctranspath"

# 外部测试集 (可选,不需要就注释掉)
EXTERNAL_CSV="/home/stat-jijianxin/PFMs/Survival_code/csv_file/tcga_survival_matched.csv"
EXTERNAL_H5="/home/stat-jijianxin/PFMs/TRIDENT/tcga_filtered/20x_512px_0px_overlap/features_ctranspath"

# ========== 模型参数 ==========
IN_DIM=768              # 特征维度 (CTransPath=768, CONCH=512)
N_CLASSES=4             # 时间区间数
DROPOUT=0.5             # Dropout比率
ACT="gelu"              # 激活函数 (relu/gelu)
MAMBA_LAYER=2           # Mamba层数

# ========== 训练参数 ==========
MAX_EPOCHS=100          # 最大训练轮数
LR=2e-4                 # 学习率
WEIGHT_DECAY=1e-3       # 权重衰减
OPTIMIZER="adamw"       # 优化器 (adam/adamw)
LOSS="nll"              # 损失函数 (cox/nll)
ALPHA_SURV=0.35          # NLL损失的alpha参数
GC=1                    # 梯度累积步数

# ========== K-Fold参数 ==========
K_FOLD=5                # 折数
FOLD=                   # 指定fold (留空=训练所有fold)
VAL_RATIO=0.15          # 验证集比例
TEST_RATIO=0.15         # 测试集比例

# ========== 早停参数 ==========
WARMUP=5                # 早停预热轮数
PATIENCE=15             # 早停耐心值
STOP_EPOCH=20           # 早停最小轮数

# ========== 其他参数 ==========
RESULTS_DIR="./results_hmu_tcga"
NUM_WORKERS=0
SEED=42

# ============================================================
# 构建命令
# ============================================================

CMD="python main.py \
    --csv_path $CSV_PATH \
    --h5_dir $H5_DIR \
    --in_dim $IN_DIM \
    --n_classes $N_CLASSES \
    --dropout $DROPOUT \
    --act $ACT \
    --mamba_layer $MAMBA_LAYER \
    --max_epochs $MAX_EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --optimizer $OPTIMIZER \
    --loss $LOSS \
    --alpha_surv $ALPHA_SURV \
    --gc $GC \
    --k_fold $K_FOLD \
    --val_ratio $VAL_RATIO \
    --test_ratio $TEST_RATIO \
    --warmup $WARMUP \
    --patience $PATIENCE \
    --stop_epoch $STOP_EPOCH \
    --results_dir $RESULTS_DIR \
    --num_workers $NUM_WORKERS \
    --seed $SEED"

# 添加外部测试集参数 (如果定义了)
if [ ! -z "$EXTERNAL_CSV" ] && [ ! -z "$EXTERNAL_H5" ]; then
    CMD="$CMD --external_csv_path $EXTERNAL_CSV --external_h5_dir $EXTERNAL_H5"
fi

# 添加fold参数 (如果指定了)
if [ ! -z "$FOLD" ]; then
    CMD="$CMD --fold $FOLD"
fi

# ============================================================
# 打印配置
# ============================================================
echo "============================================================"
echo "Training Configuration"
echo "============================================================"
echo "CSV Path: $CSV_PATH"
echo "H5 Dir: $H5_DIR"
if [ ! -z "$EXTERNAL_CSV" ]; then
    echo "External CSV: $EXTERNAL_CSV"
    echo "External H5: $EXTERNAL_H5"
fi
echo ""
echo "Model: Mamba2MIL"
echo "  - Input Dim: $IN_DIM"
echo "  - N Classes: $N_CLASSES"
echo "  - Dropout: $DROPOUT"
echo "  - Activation: $ACT"
echo "  - Mamba Layers: $MAMBA_LAYER"
echo ""
echo "Training:"
echo "  - Max Epochs: $MAX_EPOCHS"
echo "  - Learning Rate: $LR"
echo "  - Weight Decay: $WEIGHT_DECAY"
echo "  - Optimizer: $OPTIMIZER"
echo "  - Loss: $LOSS"
echo ""
echo "K-Fold: $K_FOLD"
if [ ! -z "$FOLD" ]; then
    echo "  - Training Fold: $FOLD"
else
    echo "  - Training All Folds"
fi
echo ""
echo "Results Dir: $RESULTS_DIR"
echo "============================================================"
echo ""

# ============================================================
# 运行训练
# ============================================================
echo "Starting training..."
echo ""

eval $CMD

echo ""
echo "============================================================"
echo "Training completed!"
echo "Results saved to: $RESULTS_DIR"
echo "============================================================"
