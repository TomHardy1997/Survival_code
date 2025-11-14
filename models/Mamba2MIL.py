"""
MambaMIL - 使用 Mamba2 + Mask支持
"""
import torch
import torch.nn as nn
from mamba_ssm import Mamba2
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


class Mamba2MIL(nn.Module):
    def __init__(self, in_dim, n_classes, dropout, act, survival=False, layer=2):
        super(Mamba2MIL, self).__init__()
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

        # 使用 Mamba2
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

        self.n_classes = n_classes
        self.classifier = nn.Linear(512, self.n_classes)
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.apply(initialize_weights)

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, n, in_dim] 输入特征
            mask: [B, n] 0表示padding位置,1表示有效位置
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # [1, n, in_dim]
        
        B, N, _ = x.shape
        h = x.float()  # [B, n, in_dim]
        
        # 如果没有提供mask,默认全部有效
        if mask is None:
            mask = torch.ones(B, N, device=x.device)
        
        # 确保mask是正确的形状
        if len(mask.shape) == 1:
            mask = mask.unsqueeze(0)  # [1, n]
        
        h = self._fc1(h)  # [B, n, 512]

        # Mamba2 forward
        for layer in self.layers:
            h_ = h
            h = layer[0](h)  # LayerNorm
            h = layer[1](h)  # Mamba2
            h = h + h_       # 残差连接
            
            # 应用mask - 将padding位置的特征置零
            h = h * mask.unsqueeze(-1)  # [B, n, 512] * [B, n, 1]

        h = self.norm(h)
        
        # 计算attention时考虑mask
        A = self.attention(h)  # [B, n, 1]
        A = A.squeeze(-1)  # [B, n]
        
        # 将padding位置的attention设为很小的负数(softmax后接近0)
        A = A.masked_fill(mask == 0, -1e9)
        
        A = F.softmax(A, dim=-1)  # [B, n]
        A = A.unsqueeze(1)  # [B, 1, n]
        
        h = torch.bmm(A, h)  # [B, 1, 512]
        h = h.squeeze(1)  # [B, 512]

        logits = self.classifier(h)  # [B, n_classes]
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        
        if self.survival:
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, A.squeeze(1), h  # 返回attention和特征用于可视化
            
        return logits, Y_prob, Y_hat, A.squeeze(1), h
    
    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fc1 = self._fc1.to(device)
        self.layers = self.layers.to(device)
        self.attention = self.attention.to(device)
        self.norm = self.norm.to(device)
        self.classifier = self.classifier.to(device)
