import torch.nn as nn
import torch.nn.functional as F


class KLLoss(nn.Module):
    """Symmetric KL divergence loss with temperature scaling."""

    def __init__(self):
        super().__init__()
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, prediction, label):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, dim=1)
        probs2 = F.softmax(label * 10, dim=1)
        return self.kl_div(probs1, probs2) * batch_size
