import torch


class NormalizedCrossCorrelation:

    def __init__(self, epsilon=1e-5):
        self.epsilon = epsilon

    def __call__(self, y_pred, y_true):
        t0 = torch.mean(y_true, [1, 2, 3], keepdim=True)
        p0 = torch.mean(y_pred, [1, 2, 3], keepdim=True)

        t2 = torch.mean(y_true**2, [1, 2, 3], keepdim=True)
        p2 = torch.mean(y_pred**2, [1, 2, 3], keepdim=True)

        t_std = torch.sqrt(t2 - t0**2)
        p_std = torch.sqrt(p2 - p0**2)

        num = (y_true - t0) * (y_pred - p0)
        den = (t_std * p_std) + self.epsilon

        return -torch.mean(num / den)


class TotalVariation:

    def __call__(self, flow):
        dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
        dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])

        d = torch.mean(dx) + torch.mean(dy)

        return d / 2.0


class L2Squared:

    def __call__(self, y_pred, y_true):
        return torch.mean(((y_true - y_pred)**2).sum(dim=1))
