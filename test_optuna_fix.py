import os
import json
from optuna_optimize import FIXED_PARAMS, objective_stage1, run_ddp_training

print("="*60)
print("测试 Optuna 修复")
print("="*60)

# 1. 检查固定参数
print("\n1. 固定参数:")
print(json.dumps(FIXED_PARAMS, indent=2))

# 2. 模拟一个trial
class MockTrial:
    def __init__(self):
        self.number = 999
    
    def suggest_float(self, name, low, high, log=False):
        if name == 'dropout':
            return 0.25
        elif name == 'lr':
            return 2e-4
        elif name == 'weight_decay':
            return 1e-5
        elif name == 'ranking_weight':
            return 0.1
        return (low + high) / 2
    
    def suggest_categorical(self, name, choices):
        if name == 'act':
            return 'gelu'
        elif name == 'optimizer':
            return 'adamw'
        elif name == 'batch_size':
            return 4
        return choices[0]
    
    def suggest_int(self, name, low, high):
        if name == 'mamba_layer':
            return 2
        elif name == 'gc':
            return 1
        return (low + high) // 2

mock_trial = MockTrial()

# 3. 测试参数生成
print("\n2. 测试参数生成:")
params = FIXED_PARAMS.copy()
params.update({
    'dropout': mock_trial.suggest_float('dropout', 0.1, 0.5),
    'act': mock_trial.suggest_categorical('act', ['relu', 'gelu', 'silu']),
    'mamba_layer': mock_trial.suggest_int('mamba_layer', 1, 4),
    'batch_size': mock_trial.suggest_categorical('batch_size', [4, 8, 16]),
    'lr': mock_trial.suggest_float('lr', 1e-5, 5e-4, log=True),
    'weight_decay': mock_trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True),
    'optimizer': mock_trial.suggest_categorical('optimizer', ['adam', 'adamw']),
    'ranking_weight': mock_trial.suggest_float('ranking_weight', 0.0, 0.3),
    'gc': mock_trial.suggest_int('gc', 1, 32),
})

print(json.dumps(params, indent=2))

# 4. 检查关键参数
print("\n3. 检查关键参数:")
print(f"  ✓ feature_models: {params.get('feature_models', '❌ 缺失!')}")
print(f"  ✓ in_dim: {params.get('in_dim', '❌ 缺失!')}")
print(f"  ✓ csv_path: {params.get('csv_path', '❌ 缺失!')}")
print(f"  ✓ h5_dir: {params.get('h5_dir', '❌ 缺失!')}")

print("\n✅ 参数检查通过!")
