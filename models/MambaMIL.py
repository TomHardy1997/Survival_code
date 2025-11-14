"""
MambaMIL
"""
import torch
import torch.nn as nn
from mamba.mamba_ssm import SRMamba
from mamba.mamba_ssm import BiMamba
from mamba.mamba_ssm import Mamba
import torch.nn.functional as F


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class MambaMIL(nn.Module):
    def __init__(self, in_dim, n_classes, dropout, act, survival = False, layer=2, rate=10, type="SRMamba"):
        super(MambaMIL, self).__init__()
        self._fc1 = [nn.Linear(in_dim, 512)]
        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]
        if dropout:
            self._fc1 += [nn.Dropout(dropout)]

        self._fc1 = nn.Sequential(*self._fc1)
        self.norm = nn.LayerNorm(512)
        self.layers = nn.ModuleList()
        self.survival = survival

        if type == "SRMamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(512),
                        SRMamba(
                            d_model=512,
                            d_state=16,  
                            d_conv=4,    
                            expand=2,
                        ),
                        )
                )
        elif type == "Mamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(512),
                        Mamba(
                            d_model=512,
                            d_state=16,  
                            d_conv=4,    
                            expand=2,
                        ),
                        )
                )
        elif type == "BiMamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(512),
                        BiMamba(
                            d_model=512,
                            d_state=16,  
                            d_conv=4,    
                            expand=2,
                        ),
                        )
                )
        else:
            raise NotImplementedError("Mamba [{}] is not implemented".format(type))

        self.n_classes = n_classes
        self.classifier = nn.Linear(512, self.n_classes)
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.rate = rate
        self.type = type

        self.apply(initialize_weights)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.expand(1, -1, -1)
        h = x.float()  # [B, n, 1024]
        
        h = self._fc1(h)  # [B, n, 256]

        if self.type == "SRMamba":
            for layer in self.layers:
                h_ = h
                h = layer[0](h)
                h = layer[1](h, rate=self.rate)
                h = h + h_
        elif self.type == "Mamba" or self.type == "BiMamba":
            for layer in self.layers:
                h_ = h
                h = layer[0](h)
                h = layer[1](h)
                h = h + h_

        h = self.norm(h)
        A = self.attention(h) # [B, n, K]
        A = torch.transpose(A, 1, 2)
        A = F.softmax(A, dim=-1) # [B, K, n]
        h = torch.bmm(A, h) # [B, K, 512]
        h = h.squeeze(0)

        logits = self.classifier(h)  # [B, n_classes]
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        A_raw = None
        results_dict = None
        if self.survival:
            Y_hat = torch.topk(logits, 1, dim = 1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, None, None
        return logits, Y_prob, Y_hat, A_raw, results_dict
    
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fc1 = self._fc1.to(device)
        self.layers  = self.layers.to(device)
        
        self.attention = self.attention.to(device)
        self.norm = self.norm.to(device)
        self.classifier = self.classifier.to(device)


if __name__ == '__main__':
    import torch
    import sys
    sys.path.append('.')  # 确保能导入 dataset
    from dataset import PrognosisDataset
    
    print("=" * 60)
    print("🧪 MambaMIL 模型测试")
    print("=" * 60)
    
    # 1. 加载数据
    csv_path = '../tcga_survival_matched.csv'
    h5_dir = '/home/stat-jijianxin/PFMs/TRIDENT/tcga_filtered/20x_512px_0px_overlap/features_conch_v15'
    
    dataset = PrognosisDataset(csv_path, h5_dir)
    print(f"\n✅ 数据集加载成功: {len(dataset)} 个样本")
    
    # 2. 获取一个样本
    sample = dataset[0]
    patient, gender, age, label, sur_time, censor, features, coords, num_patches = sample
    
    print(f"\n样本信息:")
    print(f"  患者: {patient}")
    print(f"  特征形状: {features.shape}")
    print(f"  Patch数量: {num_patches}")
    print(f"  真实标签: {label}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")
    
    # 3. 测试 SRMamba
    print("\n" + "-" * 60)
    print("测试 SRMamba")
    print("-" * 60)
    
    model1 = MambaMIL(
        in_dim=features.shape[1],  # 768
        n_classes=4,
        dropout=0.25,
        act='gelu',
        survival=False,
        layer=2,
        rate=10,
        type="SRMamba"
    ).to(device)
    
    print(f"✅ SRMamba 模型创建成功")
    print(f"  参数量: {sum(p.numel() for p in model1.parameters()):,}")
    
    model1.eval()
    features_gpu = features.to(device)
    
    with torch.no_grad():
        logits, Y_prob, Y_hat, _, _ = model1(features_gpu)
    
    print(f"\n✅ 前向传播成功")
    print(f"  预测类别: {Y_hat.item()}")
    print(f"  真实标签: {label}")
    print(f"  预测概率: {Y_prob.cpu().numpy()}")
    print(f"  Logits: {logits.cpu().numpy()}")
    
    # 4. 测试 Mamba
    print("\n" + "-" * 60)
    print("测试 Mamba")
    print("-" * 60)
    
    model2 = MambaMIL(
        in_dim=features.shape[1],
        n_classes=4,
        dropout=0.25,
        act='gelu',
        survival=False,
        layer=2,
        type="Mamba"
    ).to(device)
    
    print(f"✅ Mamba 模型创建成功")
    print(f"  参数量: {sum(p.numel() for p in model2.parameters()):,}")
    
    model2.eval()
    with torch.no_grad():
        logits2, Y_prob2, Y_hat2, _, _ = model2(features_gpu)
    
    print(f"\n✅ 前向传播成功")
    print(f"  预测类别: {Y_hat2.item()}")
    print(f"  预测概率: {Y_prob2.cpu().numpy()}")
    
    # 5. 测试 BiMamba
    print("\n" + "-" * 60)
    print("测试 BiMamba")
    print("-" * 60)
    
    model3 = MambaMIL(
        in_dim=features.shape[1],
        n_classes=4,
        dropout=0.25,
        act='gelu',
        survival=False,
        layer=2,
        type="BiMamba"
    ).to(device)
    
    print(f"✅ BiMamba 模型创建成功")
    print(f"  参数量: {sum(p.numel() for p in model3.parameters()):,}")
    
    model3.eval()
    with torch.no_grad():
        logits3, Y_prob3, Y_hat3, _, _ = model3(features_gpu)
    
    print(f"\n✅ 前向传播成功")
    print(f"  预测类别: {Y_hat3.item()}")
    print(f"  预测概率: {Y_prob3.cpu().numpy()}")
    
    # 6. 测试生存分析模式
    print("\n" + "-" * 60)
    print("测试生存分析模式")
    print("-" * 60)
    
    model_surv = MambaMIL(
        in_dim=features.shape[1],
        n_classes=4,
        dropout=0.25,
        act='gelu',
        survival=True,  # 开启生存分析
        layer=2,
        type="SRMamba"
    ).to(device)
    
    print(f"✅ 生存分析模型创建成功")
    
    model_surv.eval()
    with torch.no_grad():
        hazards, S, Y_hat_surv, _, _ = model_surv(features_gpu)
    
    print(f"\n✅ 前向传播成功")
    print(f"  Hazards 形状: {hazards.shape}")
    print(f"  Survival 形状: {S.shape}")
    print(f"  预测类别: {Y_hat_surv.item()}")
    print(f"  Hazards: {hazards.cpu().numpy()}")
    print(f"  Survival: {S.cpu().numpy()}")
    
    # 7. 测试反向传播
    print("\n" + "-" * 60)
    print("测试反向传播")
    print("-" * 60)
    
    model1.train()
    logits, _, _, _, _ = model1(features_gpu)
    target = torch.tensor([label]).to(device)
    loss = F.cross_entropy(logits, target)
    loss.backward()
    
    print(f"✅ 反向传播成功")
    print(f"  损失值: {loss.item():.4f}")
    
    print("\n" + "=" * 60)
    print("🎉 所有测试完成!")
    print("=" * 60)
