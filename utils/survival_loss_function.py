# survival_loss_function.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CoxSurvLoss(nn.Module):
    """
    Cox比例风险损失函数
    
    Cox Proportional Hazards Loss for survival analysis.
    适用于离散时间生存分析模型。
    
    Args:
        None
        
    Forward Args:
        hazards: torch.Tensor [batch, n_classes] 
            每个时间区间的风险概率
        S: torch.Tensor [batch, n_classes]
            生存函数（未使用，保持接口一致性）
        Y: torch.Tensor [batch]
            离散化的生存时间区间索引 (0, 1, 2, ..., n_classes-1)
        c: torch.Tensor [batch]
            删失标记 (1=删失/censored, 0=事件发生/event)
            
    Returns:
        loss: torch.Tensor 标量损失值
        
    Example:
        >>> criterion = CoxSurvLoss()
        >>> hazards = torch.rand(32, 4)  # batch=32, n_classes=4
        >>> S = torch.rand(32, 4)
        >>> Y = torch.randint(0, 4, (32,))
        >>> c = torch.randint(0, 2, (32,))
        >>> loss = criterion(hazards, S, Y, c)
    """
    
    def __init__(self):
        super(CoxSurvLoss, self).__init__()

    def forward(
        self, 
        hazards: torch.Tensor, 
        S: torch.Tensor, 
        Y: torch.Tensor, 
        c: torch.Tensor, 
        **kwargs
    ) -> torch.Tensor:
        batch_size = len(Y)
        Y = Y.view(batch_size, 1)  # [batch, 1]
        c = c.view(batch_size, 1).float()  # [batch, 1]
        
        # 创建 one-hot 编码
        Y_one_hot = torch.zeros_like(hazards).scatter_(1, Y, 1)
        
        # 计算对数风险（添加小值避免log(0)）
        hazards_log = torch.log(torch.clamp(hazards, min=1e-7))
        
        # Cox loss: 只对未删失样本计算到该时间点的累积对数风险
        # 删失样本不贡献损失（因为我们不知道真实事件时间）
        loss = -(1 - c) * (Y_one_hot * hazards_log).sum(dim=1)
        loss = loss.mean()
        
        return loss


class NLLSurvLoss(nn.Module):
    """
    负对数似然生存损失函数
    
    Negative Log-Likelihood Survival Loss for discrete-time survival analysis.
    
    该损失函数基于离散时间生存分析，将连续时间离散化为k个区间：
    T_cont ∈ {[0, a_1), [a_1, a_2), ..., [a_(k-1), ∞)}
    
    核心概念：
    - hazards(t): P(Y=t | Y≥t, X) - 在存活到时间t的条件下，在时间t死亡的概率
    - S(t): P(Y > t | X) - 存活超过时间t的概率
    - S(t) = ∏(1 - hazards(i)) for i=0 to t
    
    Args:
        alpha: float, default=0.0
            平衡删失和未删失样本的权重参数
            - alpha=0: 删失和未删失样本权重相同
            - alpha>0: 给未删失样本更多权重
            - alpha=1: 只考虑未删失样本
            
    Forward Args:
        hazards: torch.Tensor [batch, n_classes]
            每个时间区间的风险概率
        S: torch.Tensor [batch, n_classes]
            生存函数，S(t) = P(Y > t | X)
        Y: torch.Tensor [batch]
            离散化的生存时间区间索引 (0, 1, 2, ..., n_classes-1)
        c: torch.Tensor [batch]
            删失标记 (1=删失/censored, 0=事件发生/event)
        alpha: float, optional
            如果提供，覆盖初始化时的alpha值
            
    Returns:
        loss: torch.Tensor 标量损失值
        
    Loss Formulation:
        对于未删失样本 (c=0):
            L = -log(P(Y=y)) = -log(S(y-1) * h(y))
            = -log(S(y-1)) - log(h(y))
            
        对于删失样本 (c=1):
            L = -log(P(Y>y)) = -log(S(y))
            
        总损失:
            L_total = (1-α) * (L_censored + L_uncensored) + α * L_uncensored
            
    Example:
        >>> criterion = NLLSurvLoss(alpha=0.15)
        >>> hazards = torch.rand(32, 4)
        >>> S = torch.cumprod(1 - hazards, dim=1)  # 计算生存函数
        >>> Y = torch.randint(0, 4, (32,))
        >>> c = torch.randint(0, 2, (32,))
        >>> loss = criterion(hazards, S, Y, c)
    """
    
    def __init__(self, alpha: float = 0.0):
        super(NLLSurvLoss, self).__init__()
        self.alpha = alpha

    def forward(
        self, 
        hazards: torch.Tensor, 
        S: torch.Tensor, 
        Y: torch.Tensor, 
        c: torch.Tensor, 
        alpha: Optional[float] = None,
        **kwargs
    ) -> torch.Tensor:
        if alpha is None:
            alpha = self.alpha
        
        batch_size = len(Y)
        Y = Y.view(batch_size, 1)  # [batch, 1]
        c = c.view(batch_size, 1).float()  # [batch, 1]
        
        # Padding: S(-1) = 1 (所有患者在时间-1都存活)
        # S_padded[0] = S(-1) = 1
        # S_padded[1] = S(0)
        # S_padded[2] = S(1), ...
        S_padded = torch.cat([torch.ones_like(c), S], dim=1)
        
        # 获取对应时间点的概率
        s_prev = torch.gather(S_padded, dim=1, index=Y).clamp(min=1e-7)      # S(Y-1): 存活到Y-1的概率
        h = torch.gather(hazards, dim=1, index=Y).clamp(min=1e-7)            # h(Y): 在Y时刻死亡的概率
        s_now = torch.gather(S_padded, dim=1, index=Y+1).clamp(min=1e-7)     # S(Y): 存活超过Y的概率
        
        # 计算负对数似然
        # 未删失样本: -log(P(Y=y)) = -log(S(y-1) * h(y)) = -log(S(y-1)) - log(h(y))
        uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h))
        
        # 删失样本: -log(P(Y>y)) = -log(S(y))
        censored_loss = -c * torch.log(s_now)
        
        # 组合损失（可选加权）
        if alpha == 0:
            # 标准NLL损失
            loss = uncensored_loss + censored_loss
        else:
            # 加权损失：给未删失样本更多权重
            loss = (1 - alpha) * (uncensored_loss + censored_loss) + alpha * uncensored_loss
        
        loss = loss.mean()
        
        return loss


class RankingLoss(nn.Module):
    """
    排序损失函数（可选）
    
    Ranking Loss for survival analysis, ensures that patients with 
    earlier event times have higher risk scores.
    
    Args:
        margin: float, default=0.0
            排序损失的边界值
            
    Forward Args:
        hazards: torch.Tensor [batch, n_classes]
        S: torch.Tensor [batch, n_classes]
        Y: torch.Tensor [batch]
        c: torch.Tensor [batch]
        
    Returns:
        loss: torch.Tensor
    """
    
    def __init__(self, margin: float = 0.0):
        super(RankingLoss, self).__init__()
        self.margin = margin
    
    def forward(
        self,
        hazards: torch.Tensor,
        S: torch.Tensor,
        Y: torch.Tensor,
        c: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        batch_size = len(Y)
        
        # 计算风险分数（可以使用累积风险或其他度量）
        risk_scores = hazards.sum(dim=1)  # [batch]
        
        # 构建成对比较
        loss = 0.0
        count = 0
        
        for i in range(batch_size):
            if c[i] == 0:  # 只考虑未删失样本
                for j in range(batch_size):
                    if Y[j] > Y[i]:  # j的事件时间晚于i
                        # i应该有更高的风险分数
                        loss += torch.relu(self.margin + risk_scores[j] - risk_scores[i])
                        count += 1
        
        if count > 0:
            loss = loss / count
        
        return loss


# ===================== 损失函数工厂 =====================
class SurvivalLossFactory:
    """
    生存分析损失函数工厂
    
    Example:
        >>> factory = SurvivalLossFactory()
        >>> criterion = factory.get_loss('nll', alpha=0.15)
        >>> loss = criterion(hazards, S, Y, c)
    """
    
    AVAILABLE_LOSSES = ['nll', 'cox', 'ranking']
    
    @staticmethod
    def get_loss(
        loss_type: str = 'nll',
        **kwargs
    ) -> nn.Module:
        """
        获取指定类型的损失函数
        
        Args:
            loss_type: str, 损失函数类型
                - 'nll': NLLSurvLoss
                - 'cox': CoxSurvLoss
                - 'ranking': RankingLoss
            **kwargs: 传递给损失函数的参数
            
        Returns:
            criterion: nn.Module 损失函数实例
            
        Raises:
            ValueError: 如果loss_type不在支持列表中
        """
        loss_type = loss_type.lower()
        
        if loss_type == 'nll':
            alpha = kwargs.get('alpha', 0.0)
            return NLLSurvLoss(alpha=alpha)
        
        elif loss_type == 'cox':
            return CoxSurvLoss()
        
        elif loss_type == 'ranking':
            margin = kwargs.get('margin', 0.0)
            return RankingLoss(margin=margin)
        
        else:
            raise ValueError(
                f"Unknown loss type: {loss_type}. "
                f"Available losses: {SurvivalLossFactory.AVAILABLE_LOSSES}"
            )
    
    @staticmethod
    def list_available_losses():
        """列出所有可用的损失函数"""
        return SurvivalLossFactory.AVAILABLE_LOSSES


# ===================== 便捷函数 =====================
def create_survival_loss(loss_type: str = 'nll', **kwargs) -> nn.Module:
    """
    创建生存分析损失函数的便捷函数
    
    Args:
        loss_type: str, 损失函数类型 ('nll', 'cox', 'ranking')
        **kwargs: 损失函数参数
        
    Returns:
        criterion: nn.Module
        
    Example:
        >>> criterion = create_survival_loss('nll', alpha=0.15)
        >>> loss = criterion(hazards, S, Y, c)
    """
    return SurvivalLossFactory.get_loss(loss_type, **kwargs)


if __name__ == '__main__':
    # 测试代码
    print("=" * 60)
    print("测试生存分析损失函数")
    print("=" * 60)
    
    # 创建模拟数据
    batch_size = 8
    n_classes = 4
    
    hazards = torch.rand(batch_size, n_classes)
    hazards = F.softmax(hazards, dim=1)  # 归一化
    S = torch.cumprod(1 - hazards, dim=1)  # 计算生存函数
    Y = torch.randint(0, n_classes, (batch_size,))
    c = torch.randint(0, 2, (batch_size,))
    
    print(f"\n输入数据:")
    print(f"  hazards shape: {hazards.shape}")
    print(f"  S shape: {S.shape}")
    print(f"  Y: {Y}")
    print(f"  c (censoring): {c}")
    
    # 测试 NLL Loss
    print("\n" + "-" * 60)
    print("测试 NLL Loss")
    print("-" * 60)
    nll_criterion = create_survival_loss('nll', alpha=0.15)
    nll_loss = nll_criterion(hazards, S, Y, c)
    print(f"NLL Loss: {nll_loss.item():.4f}")
    
    # 测试 Cox Loss
    print("\n" + "-" * 60)
    print("测试 Cox Loss")
    print("-" * 60)
    cox_criterion = create_survival_loss('cox')
    cox_loss = cox_criterion(hazards, S, Y, c)
    print(f"Cox Loss: {cox_loss.item():.4f}")
    
    # 测试 Ranking Loss
    print("\n" + "-" * 60)
    print("测试 Ranking Loss")
    print("-" * 60)
    ranking_criterion = create_survival_loss('ranking', margin=0.1)
    ranking_loss = ranking_criterion(hazards, S, Y, c)
    print(f"Ranking Loss: {ranking_loss.item():.4f}")
    
    # 列出所有可用损失
    print("\n" + "-" * 60)
    print("可用的损失函数:")
    print("-" * 60)
    for loss_name in SurvivalLossFactory.list_available_losses():
        print(f"  - {loss_name}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)