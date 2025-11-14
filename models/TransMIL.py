import numpy as np
import torch
import torch.nn as nn
from math import ceil
from einops import rearrange, reduce
from torch import nn, einsum
import torch.nn.functional as F


# helper functions
def exists(val):
    return val is not None


def moore_penrose_iter_pinv(x, iters = 6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim = -1)
    row = abs_x.sum(dim = -2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device = device)
    I = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z

# main attention class
class NystromAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        num_landmarks = 256,
        pinv_iterations = 6,
        residual = True,
        residual_conv_kernel = 33,
        eps = 1e-8,
        dropout = 0.
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)

    def forward(self, x, mask = None, return_attn = False):
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps

        # pad so that sequence can be evenly divided into m landmarks

        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value = 0)

            if exists(mask):
                mask = F.pad(mask, (padding, 0), value = False)

        # derive query, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # set masked positions to 0 in queries, keys, values

        if exists(mask):
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q = q * self.scale

        # generate landmarks by sum reduction, and then calculate mean using the mask

        l = ceil(n / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        q_landmarks = reduce(q, landmark_einops_eq, 'sum', l = l)
        k_landmarks = reduce(k, landmark_einops_eq, 'sum', l = l)

        # calculate landmark mask, and also get sum of non-masked elements in preparation for masked mean

        divisor = l
        if exists(mask):
            mask_landmarks_sum = reduce(mask, '... (n l) -> ... n', 'sum', l = l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        # masked mean (if mask exists)

        q_landmarks /= divisor
        k_landmarks /= divisor

        # similarities

        einops_eq = '... i d, ... j d -> ... i j'
        sim1 = einsum(einops_eq, q, k_landmarks)
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = einsum(einops_eq, q_landmarks, k)

        # masking

        if exists(mask):
            mask_value = -torch.finfo(q.dtype).max
            sim1.masked_fill_(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim3.masked_fill_(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)

        # eq (15) in the paper and aggregate values

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim = -1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)

        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        # add depth-wise conv residual of values

        if self.residual:
            out += self.res_conv(v)

        # merge and combine heads

        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)
        out = out[:, -n:]

        if return_attn:
            attn = attn1 @ attn2_inv @ attn3
            return out, attn

        return out

# transformer

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class Nystromformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_landmarks = 256,
        pinv_iterations = 6,
        attn_values_residual = True,
        attn_values_residual_conv_kernel = 33,
        attn_dropout = 0.,
        ff_dropout = 0.   
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, NystromAttention(dim = dim, dim_head = dim_head, heads = heads, num_landmarks = num_landmarks, pinv_iterations = pinv_iterations, residual = attn_values_residual, residual_conv_kernel = attn_values_residual_conv_kernel, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim = dim, dropout = ff_dropout))
            ]))

    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask) + x
            x = ff(x) + x
        return x


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,  # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,  # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, in_dim, n_classes, dropout, act, survival = False):
        super(TransMIL, self).__init__()
        ### 
        self._fc1 = [nn.Linear(in_dim, 512)]
        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]
        if dropout:
            self._fc1 += [nn.Dropout(dropout)]
            print("dropout: ", dropout)
        self._fc1 = nn.Sequential(*self._fc1)

        self.pos_layer = PPEG(dim=512)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self.classifier = nn.Linear(512, self.n_classes)

        self.apply(initialize_weights)
        self.survival = survival

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.expand(1, -1, -1)

        h = x.float()  # [B, n, 1024]

        h = self._fc1(h)  # [B, n, 256]

        # ---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 256]

        # ---->cls_token
        cls_tokens = self.cls_token.expand(1, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 256]

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 256]

        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 256]

        # ---->cls_token
        h = self.norm(h)[:, 0]

        # ---->predict
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
        self.pos_layer = self.pos_layer.to(device)
        self.layer1 = self.layer1.to(device)
        self.layer2 = self.layer2.to(device)
        self.norm = self.norm.to(device)
        self.classifier = self.classifier.to(device)









if __name__ == "__main__":
    import sys
    sys.path.append('.')
    from dataset import PrognosisDataset
    
    print("=" * 60)
    print("🧪 TransMIL 模型测试")
    print("=" * 60)
    
    # 1. 测试随机数据
    print("\n" + "-" * 60)
    print("测试随机数据")
    print("-" * 60)
    
    data = torch.randn((1, 6000, 1024)).cuda()
    model_random = TransMIL(in_dim=1024, n_classes=4, act='gelu', dropout=0.25)
    model_random = model_random.cuda()
    
    print(f"✅ 模型创建成功")
    print(f"  参数量: {sum(p.numel() for p in model_random.parameters()):,}")
    
    model_random.eval()
    with torch.no_grad():
        logits, Y_prob, Y_hat, _, _ = model_random(data)
    
    print(f"\n✅ 随机数据测试成功")
    print(f"  输入形状: {data.shape}")
    print(f"  预测类别: {Y_hat.item()}")
    print(f"  预测概率: {Y_prob.cpu().numpy()}")
    
    # 2. 测试真实数据
    print("\n" + "-" * 60)
    print("测试真实数据")
    print("-" * 60)
    
    csv_path = '../tcga_survival_matched.csv'
    h5_dir = '/home/stat-jijianxin/PFMs/TRIDENT/tcga_filtered/20x_512px_0px_overlap/features_conch_v15'
    
    dataset = PrognosisDataset(csv_path, h5_dir)
    print(f"\n✅ 数据集加载成功: {len(dataset)} 个样本")
    
    sample = dataset[0]
    patient, gender, age, label, sur_time, censor, features, coords, num_patches = sample
    
    print(f"\n样本信息:")
    print(f"  患者: {patient}")
    print(f"  特征形状: {features.shape}")
    print(f"  Patch数量: {num_patches}")
    print(f"  真实标签: {label}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 3. 创建模型（分类模式）
    print("\n" + "-" * 60)
    print("测试分类模式")
    print("-" * 60)
    
    model = TransMIL(
        in_dim=features.shape[1],  # 768
        n_classes=4,
        dropout=0.25,
        act='gelu',
        survival=False
    ).to(device)
    
    print(f"✅ TransMIL 模型创建成功")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    model.eval()
    features_gpu = features.to(device)
    
    with torch.no_grad():
        logits, Y_prob, Y_hat, _, _ = model(features_gpu)
    
    print(f"\n✅ 前向传播成功")
    print(f"  预测类别: {Y_hat.item()}")
    print(f"  真实标签: {label}")
    print(f"  预测概率: {Y_prob.cpu().numpy()}")
    print(f"  预测{'正确' if Y_hat.item() == label else '错误'}!")
    
    # 4. 测试生存分析模式
    print("\n" + "-" * 60)
    print("测试生存分析模式")
    print("-" * 60)
    
    model_surv = TransMIL(
        in_dim=features.shape[1],
        n_classes=4,
        dropout=0.25,
        act='gelu',
        survival=True
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
    
    # 5. 测试反向传播
    print("\n" + "-" * 60)
    print("测试反向传播")
    print("-" * 60)
    
    model.train()
    logits, _, _, _, _ = model(features_gpu)
    target = torch.tensor([label]).to(device)
    loss = F.cross_entropy(logits, target)
    loss.backward()
    
    print(f"✅ 反向传播成功")
    print(f"  损失值: {loss.item():.4f}")
    
    # 6. 测试不同大小的输入
    print("\n" + "-" * 60)
    print("测试不同大小的输入")
    print("-" * 60)
    
    model.eval()
    test_sizes = [100, 500, 1000, 2000]
    
    for size in test_sizes:
        test_input = torch.randn(1, size, features.shape[1]).to(device)
        with torch.no_grad():
            logits, _, Y_hat, _, _ = model(test_input)
        print(f"  输入大小 {size:4d}: 预测={Y_hat.item()}, ✓")
    
    # 7. 测试多个样本
    print("\n" + "-" * 60)
    print("测试多个样本")
    print("-" * 60)
    
    n_samples = min(5, len(dataset))
    correct = 0
    total = 0
    
    for idx in range(n_samples):
        sample = dataset[idx]
        if sample is None:
            continue
        
        patient, gender, age, label, sur_time, censor, features, coords, num_patches = sample
        features = features.to(device)
        
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, _ = model(features)
        
        pred = Y_hat.item()
        is_correct = (pred == label)
        correct += int(is_correct)
        total += 1
        
        print(f"  样本 {idx+1}: 患者={patient}, 预测={pred}, 真实={label}, {'✓' if is_correct else '✗'}")
    
    accuracy = correct / total if total > 0 else 0
    print(f"\n准确率: {correct}/{total} = {accuracy:.2%}")
    
    print("\n" + "=" * 60)
    print("🎉 所有测试完成!")
    print("=" * 60)