from data import test_dataloader
from Utils.metrics import *
from Utils.option import *


def _valid(backbone_model, trainLogger):
    device = args.device
    test_data_set = test_dataloader(args.testset_path, batch_size=1)

    backbone_model.eval()
    ssims_1, psnrs_1 = [], []

    for idx, data in enumerate(test_data_set):
        input_img, label_img, name = data
        input_img = input_img.to(device)
        label_img = label_img.to(device)
        with torch.no_grad():
            prompt_img = backbone_model(input_img)[2]
            pred = backbone_model(input_img, prompt_img)[2]

            pred_mean = pred.float().mean()
            label_img_mean = label_img.float().mean()
            pred_adjust = torch.clamp((pred * (label_img_mean / pred_mean)), 0, 1)

        per_psnr_1 = psnr(pred_adjust, label_img)
        per_ssim_1 = ssim(pred_adjust, label_img).item()

        # per_psnr_1 = psnr(pred, label_img)
        # per_ssim_1 = ssim(pred, label_img).item()
        psnrs_1.append(per_psnr_1)
        ssims_1.append(per_ssim_1)

    return np.mean(psnrs_1), np.mean(ssims_1)
