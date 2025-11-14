# https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py
# https://arxiv.org/pdf/1802.04712.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m,nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class DAttention(nn.Module):
    def __init__(self, in_dim, n_classes, dropout, act, survival = False):
        super(DAttention, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1
        self.feature = [nn.Linear(in_dim, 512)]
        self.survival = survival
        
        if act.lower() == 'gelu':
            self.feature += [nn.GELU()]
        else:
            self.feature += [nn.ReLU()]

        if dropout:
            self.feature += [nn.Dropout(0.25)]

        self.feature = nn.Sequential(*self.feature)

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, n_classes),
        )


        # self.apply(initialize_weights)


    def forward(self, x):
        feature = self.feature(x)
        feature = feature.squeeze()
        A = self.attention(feature)
        A = torch.transpose(A, -1, -2)  # KxN
        A_raw = A
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.mm(A, feature)  # KxL
        
        logits = self.classifier(M)
        
        '''
        Survival layer
        '''
        if self.survival:
            Y_hat = torch.topk(logits, 1, dim = 1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, None, None 
        
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        
        # keep the same API with the clam
        return logits, Y_prob, Y_hat, A_raw, {}
    
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature = self.feature.to(device)
        self.attention = self.attention.to(device)
        self.classifier = self.classifier.to(device)



class GatedAttention(nn.Module):
    def __init__(self, in_dim, n_classes, dropout, act, survival = False):
        super(GatedAttention, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1
        self.feature = [nn.Linear(in_dim, 512)]
        self.survival = survival
        if act.lower() == 'gelu':
            self.feature += [nn.GELU()]
        else:
            self.feature += [nn.ReLU()]

        if dropout:
            self.feature += [nn.Dropout(0.25)]

        self.feature = nn.Sequential(*self.feature)

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, n_classes),
        )

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.squeeze()

        A_V = self.attention_V(feature)  # NxD
        A_U = self.attention_U(feature)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, feature)  # KxL

        logits = self.classifier(M)
        
        '''
        Survival layer
        '''
        if self.survival:
            Y_hat = torch.topk(logits, 1, dim = 1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, None, None 
        
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        
        # keep the same API with the clam
        return logits, Y_prob, Y_hat, None, {}





if __name__ == '__main__':
    import torch
    from dataset import PrognosisDataset
    import ipdb;ipdb.set_trace()
    print("=" * 60)
    print("🧪 简单测试")
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
    print(f"  真实标签: {label}")
    
    # 3. 测试 DAttention
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DAttention(
        in_dim=features.shape[1],  # 768
        n_classes=4,
        dropout=True,
        act='gelu',
        survival=False
    ).to(device)
    
    print(f"\n✅ DAttention 模型创建成功")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. 前向传播
    model.eval()
    features = features.to(device)
    
    with torch.no_grad():
        logits, Y_prob, Y_hat, A_raw, _ = model(features)
    
    print(f"\n✅ 前向传播成功")
    print(f"  预测类别: {Y_hat.item()}")
    print(f"  真实标签: {label}")
    print(f"  预测概率: {Y_prob.cpu().numpy()}")
    print(f"  注意力形状: {A_raw.shape if A_raw is not None else 'None'}")
    
    # 5. 测试 GatedAttention
    model2 = GatedAttention(
        in_dim=features.shape[1],
        n_classes=4,
        dropout=True,
        act='gelu',
        survival=False
    ).to(device)
    
    print(f"\n✅ GatedAttention 模型创建成功")
    print(f"  参数量: {sum(p.numel() for p in model2.parameters()):,}")
    
    with torch.no_grad():
        logits2, Y_prob2, Y_hat2, _, _ = model2(features)
    
    print(f"\n✅ 前向传播成功")
    print(f"  预测类别: {Y_hat2.item()}")
    print(f"  预测概率: {Y_prob2.cpu().numpy()}")
    
    print("\n" + "=" * 60)
    print("🎉 测试完成!")
    print("=" * 60)

