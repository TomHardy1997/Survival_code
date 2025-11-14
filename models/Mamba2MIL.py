"""
MambaMIL - 使用 Mamba2
"""
import torch
import torch.nn as nn
from mamba_ssm import Mamba2  # 改用 Mamba2
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
                        d_state=256,      # Mamba2 推荐使用更大的 state
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

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.expand(1, -1, -1)
        h = x.float()  # [B, n, in_dim]
        
        h = self._fc1(h)  # [B, n, 512]

        # Mamba2 forward
        for layer in self.layers:
            h_ = h
            h = layer[0](h)  # LayerNorm
            h = layer[1](h)  # Mamba2
            h = h + h_       # 残差连接

        h = self.norm(h)
        A = self.attention(h)  # [B, n, 1]
        A = torch.transpose(A, 1, 2)  # [B, 1, n]
        A = F.softmax(A, dim=-1)
        h = torch.bmm(A, h)  # [B, 1, 512]
        h = h.squeeze(1)  # [B, 512]

        logits = self.classifier(h)  # [B, n_classes]
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        
        if self.survival:
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, None, None
            
        return logits, Y_prob, Y_hat, None, None
    
    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fc1 = self._fc1.to(device)
        self.layers = self.layers.to(device)
        self.attention = self.attention.to(device)
        self.norm = self.norm.to(device)
        self.classifier = self.classifier.to(device)


if __name__ == "__main__":
    pass