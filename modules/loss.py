import torch
import torch.nn.functional as F


def entropy(predictions: torch.Tensor, reduction="none"):
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    if reduction == 'mean':
        return H.mean()
    else:
        return H


def cross_entropy_soft(output: torch.Tensor, target: torch.Tensor, DEVICE: torch.device, epsilon: float = 0.1, reduction: str = 'mean'):
    log_prob = F.log_softmax(output, 1)
    target = torch.zeros(output.size()).scatter_(
        1, target.unsqueeze(1).cpu(), 1).to(DEVICE)
    num_classes = output.size()[1]
    target = (1-epsilon)*target+epsilon/num_classes
    loss = (- target * log_prob).sum(dim=1)
    if reduction=='mean':
        loss = loss.mean()
    else:
        loss = loss.sum()
    return loss

