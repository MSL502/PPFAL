import torch


class LossNetwork(torch.nn.Module):
    def __init__(self, device):
        super(LossNetwork, self).__init__()
        self.L1 = torch.nn.L1Loss().to(device)

    def forward(self, pred, gt):
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        pred_fft = torch.stack((pred_fft.real, pred_fft.imag), -1)
        gt_fft = torch.fft.fft2(gt, dim=(-2, -1))
        gt_fft = torch.stack((gt_fft.real, gt_fft.imag), -1)

        fft_loss = self.L1(pred_fft, gt_fft)

        return fft_loss