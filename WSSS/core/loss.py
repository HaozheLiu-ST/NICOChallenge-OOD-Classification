import torch
import torch.nn as nn
import torch.nn.functional as F


def cos_simi(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    sim = torch.matmul(embedded_fg, embedded_bg.T)

    return torch.clamp(sim, min=0.0005, max=0.9995)


def cos_distance(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    sim = torch.matmul(embedded_fg, embedded_bg.T)

    return 1 - sim


def l2_distance(embedded_fg, embedded_bg):
    N, C = embedded_fg.size()

    # embedded_fg = F.normalize(embedded_fg, dim=1)
    # embedded_bg = F.normalize(embedded_bg, dim=1)

    embedded_fg = embedded_fg.unsqueeze(1).expand(N, N, C)
    embedded_bg = embedded_bg.unsqueeze(0).expand(N, N, C)

    return torch.pow(embedded_fg - embedded_bg, 2).sum(2) / C

# Minimize Similarity, e.g., push representation of foreground and background apart.
class SimMinLoss(nn.Module):
    def __init__(self, metric='cos', reduction='mean'):
        super(SimMinLoss, self).__init__()
        self.metric = metric
        self.reduction = reduction
        self.m = 0.25

    def forward(self, embedded_bg, embedded_fg):
        """
        :param embedded_fg: [N, C]
        :param embedded_bg: [N, C]
        :return:
        """
        if self.metric == 'l2':
            loss = self.m - l2_distance(embedded_bg, embedded_fg)
            mask = torch.where(loss < 0, torch.zeros_like(loss), torch.ones_like(loss))
            loss = mask * loss
        elif self.metric == 'cos':
            sim = cos_simi(embedded_bg, embedded_fg)
            loss = -torch.log(1 - sim)
        else:
            raise NotImplementedError

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)


# Maximize Similarity, e.g., pull representation of background and background together.
class SimMaxLoss(nn.Module):
    def __init__(self, metric='cos', alpha=0.25, reduction='mean'):
        super(SimMaxLoss, self).__init__()
        self.metric = metric
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, embedded_bg):
        """
        :param embedded_fg: [N, C]
        :param embedded_bg: [N, C]
        :return:
        """
        if self.metric == 'l2':
            loss = l2_distance(embedded_bg, embedded_bg)
            _, indices = loss.sort(descending=True, dim=1)
            _, rank = indices.sort(dim=1)
            rank = rank - 1
            rank_weights = torch.exp(-rank.float() * self.alpha)
            # mask = torch.where(loss > self.t, torch.ones_like(loss), torch.ones_like(loss))
            loss = loss * rank_weights# * mask

        elif self.metric == 'cos':
            sim = cos_simi(embedded_bg, embedded_bg)
            loss = -torch.log(sim)
            loss[loss < 0] = 0
            _, indices = sim.sort(descending=True, dim=1)
            _, rank = indices.sort(dim=1)
            rank = rank - 1
            rank_weights = torch.exp(-rank.float() * self.alpha)
            loss = loss * rank_weights
        else:
            raise NotImplementedError

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)

# Maximize Similarity, e.g., pull representation of background and background together.
# with gt label and background label
class SimMaxLossv2(nn.Module):
    def __init__(self, metric='cos', alpha=0.25, reduction='mean', num_cls=60):
        super(SimMaxLossv2, self).__init__()
        self.metric = metric
        self.alpha = alpha
        self.reduction = reduction
        self.num_cls = num_cls

    def _to_onehot(self, label):
        onehot = torch.zeros((label.shape[0], self.num_cls))
        idx = (torch.tensor([i for i in range(label.shape[0])]), label)
        onehot[idx] = 1
        return onehot

    def forward(self, embedded_bg, label):
        """
        :param embedded_fg: [N, C]
        :param embedded_bg: [N, C]
        :return:
        """
        onehot = self._to_onehot(label).cuda()
        valid = onehot @ onehot.T
        # print(label)
        # print(onehot)
        # import time
        # time.sleep(100)
        if self.metric == 'l2':
            loss = l2_distance(embedded_bg, embedded_bg)
            _, indices = loss.sort(descending=True, dim=1)
            _, rank = indices.sort(dim=1)
            rank = rank - 1
            rank_weights = torch.exp(-rank.float() * self.alpha)
            mask = torch.where(loss > self.t, torch.ones_like(loss), torch.ones_like(loss))
            loss = loss * rank_weights * mask

        elif self.metric == 'cos':
            sim = cos_simi(embedded_bg, embedded_bg)
            loss = -torch.log(sim)
            loss[loss < 0] = 0
            # print(loss.shape, onehot.shape)
            loss = loss * valid
        else:
            raise NotImplementedError

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)


if __name__ == '__main__':
    fg_embedding = torch.randn((4, 12))
    bg_embedding = torch.randn((4, 12))
    # print(fg_embedding, bg_embedding)

    # neg_contrast = NegContrastiveLoss(metric='cos')
    # neg_loss = neg_contrast(fg_embedding, bg_embedding)
    # print(neg_loss)

    # pos_contrast = PosContrastiveLoss(metric='cos')
    # pos_loss = pos_contrast(fg_embedding)
    # print(pos_loss)

    examplar = torch.tensor([[1, 2, 3, 4], [2, 3, 1, 4], [4, 2, 1, 3]])

    _, indices = examplar.sort(descending=True, dim=1)
    print(indices)
    _, rank = indices.sort(dim=1)
    print(rank)
    rank_weights = torch.exp(-rank.float() * 0.25)
    print(rank_weights)
