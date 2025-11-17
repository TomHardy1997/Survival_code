"""
MambaMIL - 使用 Mamba2 + Mask支持
支持生存分析和分类任务
"""
import torch
import torch.nn as nn
from mamba_ssm import Mamba2
import torch.nn.functional as F


def initialize_weights(module):
    """初始化模型权重"""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class Mamba2MIL(nn.Module):
    """
    Mamba2-based Multiple Instance Learning模型
    
    Args:
        in_dim: 输入特征维度
        n_classes: 类别数量
        dropout: dropout比例
        act: 激活函数 ('relu' or 'gelu')
        survival: 是否用于生存分析
        layer: Mamba2层数
        use_clinical: 是否使用临床特征(性别、年龄)
    """
    def __init__(self, 
                in_dim=512, 
                n_classes=4, 
                dropout=0.25, 
                act='relu', 
                survival=False, 
                layer=2,
                use_clinical=False):
        super(Mamba2MIL, self).__init__()
        
        self.survival = survival
        self.n_classes = n_classes
        self.use_clinical = use_clinical
        
        # 特征投影层
        self._fc1 = [nn.Linear(in_dim, 512)]
        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]
        if dropout:
            self._fc1 += [nn.Dropout(dropout)]
        self._fc1 = nn.Sequential(*self._fc1)
        
        # LayerNorm
        self.norm = nn.LayerNorm(512)
        
        # Mamba2层
        self.layers = nn.ModuleList()
        for _ in range(layer):
            self.layers.append(
                nn.Sequential(
                    nn.LayerNorm(512),
                    Mamba2(
                        d_model=512,
                        d_state=256,
                        d_conv=4,    
                        expand=4,
                    ),
                )
            )
        
        # Attention层
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # 临床特征融合
        if use_clinical:
            # 性别: 2类 (0/1), 年龄: 1个连续值
            self.clinical_fc = nn.Sequential(
                nn.Linear(3, 64),  # gender(1) + age(1) -> one-hot后变成 2+1=3
                nn.ReLU(),
                nn.Dropout(dropout if dropout else 0),
                nn.Linear(64, 128)
            )
            classifier_input_dim = 512 + 128
        else:
            classifier_input_dim = 512
        
        # 分类器
        self.classifier = nn.Linear(classifier_input_dim, self.n_classes)
        
        # 初始化权重
        self.apply(initialize_weights)

    def forward(self, x, mask=None, gender=None, age=None, coords=None):
        """
        Args:
            x: [B, n, in_dim] 输入特征
            mask: [B, n] 0表示padding位置,1表示有效位置
            gender: [B] 性别 (0或1)
            age: [B] 年龄
            coords: [B, n, 2] 坐标 (可选,暂不使用)
        
        Returns:
            如果survival=True:
                hazards: [B, n_classes] 风险
                S: [B, n_classes] 生存概率
                Y_hat: [B, 1] 预测类别
                A: [B, n] attention权重
                h: [B, 512 or 512+128] 特征向量
            否则:
                logits: [B, n_classes]
                Y_prob: [B, n_classes]
                Y_hat: [B, 1]
                A: [B, n]
                h: [B, 512 or 512+128]
        """
        # 处理输入维度
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # [1, n, in_dim]
        
        B, N, _ = x.shape
        h = x.float()  # [B, n, in_dim]
        
        # 如果没有提供mask,默认全部有效
        if mask is None:
            mask = torch.ones(B, N, device=x.device, dtype=torch.float)
        
        # 确保mask是正确的形状
        if len(mask.shape) == 1:
            mask = mask.unsqueeze(0)  # [1, n]
        
        # 特征投影
        h = self._fc1(h)  # [B, n, 512]

        # Mamba2层 + 残差连接
        for layer in self.layers:
            h_ = h
            h = layer[0](h)  # LayerNorm
            h = layer[1](h)  # Mamba2
            h = h + h_       # 残差连接
            
            # 应用mask - 将padding位置的特征置零
            h = h * mask.unsqueeze(-1)  # [B, n, 512] * [B, n, 1]

        h = self.norm(h)  # [B, n, 512]
        
        # 计算attention (考虑mask)
        A = self.attention(h)  # [B, n, 1]
        A = A.squeeze(-1)  # [B, n]
        
        # 将padding位置的attention设为很小的负数(softmax后接近0)
        A = A.masked_fill(mask == 0, -1e9)
        A = F.softmax(A, dim=-1)  # [B, n]
        
        # Attention pooling
        A_expanded = A.unsqueeze(1)  # [B, 1, n]
        h = torch.bmm(A_expanded, h)  # [B, 1, 512]
        h = h.squeeze(1)  # [B, 512]
        
        # 融合临床特征
        if self.use_clinical and gender is not None and age is not None:
            # 处理性别 - one-hot编码
            gender_onehot = F.one_hot(gender.long(), num_classes=2).float()  # [B, 2]
            
            # 归一化年龄
            age_normalized = age.unsqueeze(-1) / 100.0  # [B, 1]
            
            # 拼接临床特征
            clinical_features = torch.cat([gender_onehot, age_normalized], dim=-1)  # [B, 3]
            clinical_h = self.clinical_fc(clinical_features)  # [B, 128]
            
            # 融合WSI特征和临床特征
            h = torch.cat([h, clinical_h], dim=-1)  # [B, 512+128]

        # 分类
        logits = self.classifier(h)  # [B, n_classes]
        
        if self.survival:
            # 生存分析输出
            hazards = torch.sigmoid(logits)  # [B, n_classes]
            S = torch.cumprod(1 - hazards, dim=1)  # [B, n_classes]
            Y_hat = torch.topk(logits, 1, dim=1)[1]  # [B, 1]
            return hazards, S, Y_hat, A, h
        else:
            # 分类输出
            Y_prob = F.softmax(logits, dim=1)  # [B, n_classes]
            Y_hat = torch.topk(logits, 1, dim=1)[1]  # [B, 1]
            return logits, Y_prob, Y_hat, A, h
    
    def relocate(self):
        """将模型移动到GPU"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        return self


# ============= 测试代码 =============
if __name__ == '__main__':
    print("="*60)
    print("Testing Mamba2MIL")
    print("="*60)
    
    # 检查CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available! Mamba2 requires CUDA.")
        print("Please run on a machine with GPU.")
        exit(1)
    
    # 测试参数
    batch_size = 4
    num_patches = 1000
    feature_dim = 512
    n_classes = 4
    
    # 创建模型
    print("\n1. Testing model without clinical features:")
    model = Mamba2MIL(
        in_dim=feature_dim,
        n_classes=n_classes,
        dropout=0.25,
        act='relu',
        survival=True,
        layer=2,
        use_clinical=False
    ).to(device)  # 移到GPU
    print(f"Model created: {sum(p.numel() for p in model.parameters())} parameters")
    
    # 创建测试数据 - 直接在GPU上创建
    x = torch.randn(batch_size, num_patches, feature_dim, device=device)
    
    # 创建mask (模拟变长序列)
    mask = torch.ones(batch_size, num_patches, device=device)
    mask[0, 800:] = 0  # 第1个样本只有800个patch
    mask[1, 600:] = 0  # 第2个样本只有600个patch
    mask[2, 900:] = 0  # 第3个样本只有900个patch
    
    print(f"\nInput shape: {x.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Valid patches: {mask.sum(dim=1).tolist()}")
    
    # 前向传播
    with torch.no_grad():  # 测试时不需要梯度
        hazards, S, Y_hat, A, h = model(x, mask=mask)
    
    print(f"\nOutput shapes:")
    print(f"  Hazards: {hazards.shape}")  # [4, 4]
    print(f"  Survival: {S.shape}")       # [4, 4]
    print(f"  Y_hat: {Y_hat.shape}")      # [4, 1]
    print(f"  Attention: {A.shape}")      # [4, 1000]
    print(f"  Features: {h.shape}")       # [4, 512]
    
    # 检查attention是否正确mask
    print(f"\nAttention sum (should be ~1.0): {A.sum(dim=1).tolist()}")
    print(f"Attention on padding (should be ~0.0):")
    print(f"  Sample 0, patch 900: {A[0, 900].item():.6f}")
    print(f"  Sample 1, patch 700: {A[1, 700].item():.6f}")
    
    # 测试带临床特征的模型
    print("\n" + "="*60)
    print("2. Testing model WITH clinical features:")
    model_clinical = Mamba2MIL(
        in_dim=feature_dim,
        n_classes=n_classes,
        dropout=0.25,
        act='relu',
        survival=True,
        layer=2,
        use_clinical=True
    ).to(device)  # 移到GPU
    print(f"Model created: {sum(p.numel() for p in model_clinical.parameters())} parameters")
    
    # 创建临床数据
    gender = torch.tensor([0, 1, 0, 1], device=device)  # Male, Female, Male, Female
    age = torch.tensor([65.0, 72.0, 58.0, 81.0], device=device)
    
    print(f"\nClinical data:")
    print(f"  Gender: {gender.tolist()}")
    print(f"  Age: {age.tolist()}")
    
    # 前向传播
    with torch.no_grad():
        hazards, S, Y_hat, A, h = model_clinical(x, mask=mask, gender=gender, age=age)
    
    print(f"\nOutput shapes:")
    print(f"  Hazards: {hazards.shape}")
    print(f"  Survival: {S.shape}")
    print(f"  Y_hat: {Y_hat.shape}")
    print(f"  Attention: {A.shape}")
    print(f"  Features: {h.shape}")  # [4, 640] = 512 + 128
    
    # 测试分类模式
    print("\n" + "="*60)
    print("3. Testing classification mode:")
    model_cls = Mamba2MIL(
        in_dim=feature_dim,
        n_classes=n_classes,
        dropout=0.25,
        act='relu',
        survival=False,  # 分类模式
        layer=2,
        use_clinical=True
    ).to(device)  # 移到GPU
    
    with torch.no_grad():
        logits, Y_prob, Y_hat, A, h = model_cls(x, mask=mask, gender=gender, age=age)
    
    print(f"\nOutput shapes:")
    print(f"  Logits: {logits.shape}")
    print(f"  Y_prob: {Y_prob.shape}")
    print(f"  Y_hat: {Y_hat.shape}")
    print(f"  Attention: {A.shape}")
    print(f"  Features: {h.shape}")
    
    print(f"\nProbability sum (should be 1.0): {Y_prob.sum(dim=1).tolist()}")
    
    # 测试梯度反向传播
    print("\n" + "="*60)
    print("4. Testing backward pass:")
    model_cls.train()
    
    # 创建一个简单的loss
    target = torch.randint(0, n_classes, (batch_size,), device=device)
    logits, Y_prob, Y_hat, A, h = model_cls(x, mask=mask, gender=gender, age=age)
    loss = F.cross_entropy(logits, target)
    
    print(f"Loss: {loss.item():.4f}")
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    has_grad = sum(1 for p in model_cls.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model_cls.parameters())
    print(f"Parameters with gradients: {has_grad}/{total_params}")
    
    print("\n" + "="*60)
    print("All tests passed! ✅")
    print("="*60)