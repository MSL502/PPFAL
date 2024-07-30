from Utils.option import *
from data import test_dataloader
from Utils.utils import *
from Utils.metrics import *
# from Utils.metrics import ssim as ssim
from torchvision import utils as vutils
from models.network import Network as Net
import lpips
from Utils.option import *
import torch
import cv2



def _eval(testLogger):

    test_datalaoder = test_dataloader(args.testset_path, batch_size=1)
    device = args.device
    loss_fn = lpips.LPIPS(net='vgg').to(device)

    backbone_model = Net(args.mode)
    backbone_model = backbone_model.to(device)
    state_dict = torch.load(args.test_model)
    backbone_model.load_state_dict(state_dict['backbone_model'])

    backbone_model = backbone_model.to(device)
    backbone_model.eval()
    torch.cuda.empty_cache()
    ssims_1, psnrs_1, lpips_1 = [], [], []

    for iter_idx, test_data in enumerate(test_datalaoder):
        input_img, label_img, name = test_data
        input_img = input_img.to(device)
        label_img = label_img.to(device)

        with torch.no_grad():

            # 1 test
            # pred = backbone_model(input_img, input_img)[2]

            # 2 test
            # print(name[0])
            prompt_img = backbone_model(input_img)[2]
            pred = backbone_model(input_img, prompt_img)[2]

            # label_img = label_img / 255.
            # pred = pred / 255.
            # mean_gray_out = cv2.cvtColor(pred.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
            # mean_gray_gt = cv2.cvtColor(label_img.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
            # normal_img_adjust = np.clip(pred * (mean_gray_gt / mean_gray_out), 0, 1)
            #
            # pred = (normal_img_adjust * 255).astype(np.uint8)
            # label_img = (gt_img * 255).astype(np.uint8)

            pred_mean = pred.float().mean()
            label_img_mean = label_img.float().mean()
            pred_adjust = torch.clamp((pred * (label_img_mean / pred_mean)), 0, 1)

            # 3 test
            # prompt_img = backbone_model(input_img)[2]
            # prompt_img = backbone_model(input_img, prompt_img)[2]
            # pred = backbone_model(input_img, prompt_img)[2]

            # 4 test
            # prompt_img = backbone_model(input_img)[2]
            # prompt_img = backbone_model(input_img, prompt_img)[2]
            # prompt_img = backbone_model(input_img, prompt_img)[2]
            # pred = backbone_model(input_img, prompt_img)[2]


            per_psnr_1 = psnr(pred_adjust, label_img)
            per_ssim_1 = ssim(pred_adjust, label_img).item()
            per_lpips_value = loss_fn(pred_adjust, label_img).item()
            # print(per_lpips_value)

            ssims_1.append(per_ssim_1)
            psnrs_1.append(per_psnr_1)
            lpips_1.append(per_lpips_value)

        print(f'\n {name[0]} iter processing:{iter_idx + 1}   psnr:{per_psnr_1:.4f}  ssim:{per_ssim_1:.4f}  lpips:{per_psnr_1:.4f}', end='',flush=True)
        testLogger.write(f'\n {name[0]} iter processing:{iter_idx + 1}   psnr:{per_psnr_1:.4f}  ssim:{per_ssim_1:.4f}  lpips:{per_psnr_1:.4f}')

        if args.save_image:
            vutils.save_image(pred, os.path.join(args.output_dir, f'{name[0]}'), normalize=True)

        avg_ssim_1 = np.mean(ssims_1)
        avg_psnr_1 = np.mean(psnrs_1)
        avg_lpips_1 = np.mean(lpips_1)

    print(f'\n------------------------------------------------------')
    testLogger.write(f'\n------------------------------------------------------')
    print(f'\navg_psnr:{avg_psnr_1:.4f}  avg_ssim:{avg_ssim_1:.4f}  avg_lpips:{avg_lpips_1:.4f}', end='', flush=True)
    testLogger.write(f'\navg_psnr:{avg_psnr_1:.4f}  avg_ssim is:{avg_ssim_1:.4f}  avg_lpips:{avg_lpips_1:.4f}')

