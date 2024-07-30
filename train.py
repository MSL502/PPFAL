from data.data_load import *
from tensorboardX import SummaryWriter
from valid import _valid
from Utils.utils import *
from Utils.option import *
from Losses.SpaLoss import LossNetwork as SpaLoss
from Losses.FALLoss import LossNetwork as FALLoss
from Losses.PerceptiveLoss import LossNetwork as PerceptiveLoss
from models.network import Network as Net
import torch
import torch.nn as nn
from warmup_scheduler.scheduler import GradualWarmupScheduler
import torch.nn.functional as F
import torchvision


def _train(trainLogger):
    start_epoch = 1
    max_psnr_1, max_ssim_1, best_psnr_epoch_1, best_ssim_epoch_1 = 0, 0, 0, 0

    train_syn_loader = train_syn_dataloader(args.trainset_path, args.batch_size)
    max_iter = len(train_syn_loader)

    device = args.device
    backbone_model = Net(args.mode)
    backbone_model = backbone_model.to(device)

    criterion = []
    criterion.append(FALLoss())
    # criterion.append(nn.L1Loss().to(device))
    criterion.append(SpaLoss(device))
    criterion.append(PerceptiveLoss(device))

    optimizer_backbone = torch.optim.Adam(backbone_model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    warmup_epochs = 3
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_backbone, T_max=args.num_epochs - warmup_epochs, eta_min=1e-08)
    scheduler = GradualWarmupScheduler(optimizer_backbone, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()

    # optimizer_backbone = torch.optim.Adam(params=filter(lambda x: x.requires_grad, backbone_model.parameters()),lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08)

    if args.resume and os.path.exists(args.resume):
        ckp = torch.load(args.resume)
        start_epoch = ckp['epoch']
        optimizer_backbone.load_state_dict(ckp['optimizer_backbone'])
        backbone_model.load_state_dict(ckp['backbone_model'])
        max_psnr_1 = ckp['max_psnr_1']
        max_ssim_1 = ckp['max_ssim_1']
        print(f'load_pretrained_model from {args.resume} Resume form epoch {start_epoch}  start_poch:{start_epoch + 1}')
        trainLogger.write(f'load_pretrained_model from {args.resume} Resume form epoch {start_epoch}  start_poch:{start_epoch + 1}')
        start_epoch = start_epoch + 1
    else:
        print('train from scratch *** ')
        trainLogger.write('train from scratch *** ')

    writer = SummaryWriter()
    epoch_FAL_adder, epoch_spa_adder, epoch_percept_adder = Adder(),Adder(),Adder()
    iter_FAL_adder, iter_spa_adder,iter_percept_adder = Adder(),Adder(),Adder()
    epoch_timer, iter_timer, total_time = Timer('m'), Timer('m'), Timer('')
    total_time.tic()
    out_img_dict = {}

    for param in backbone_model.parameters():
        param.requires_grad = True

    for epoch_idx in range(start_epoch, args.num_epochs + 1):

        epoch_timer.tic()
        iter_timer.tic()
        backbone_model.train()

        # lr = adjust_learning_rate(epoch_idx, optimizer_backbone)
        for iter_idx, data in enumerate(train_syn_loader):

            optimizer_backbone.zero_grad()

            input_img, label_img, img_name_list = data
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            # stage 1
            prompt_img = backbone_model(input_img)[2]
            # stage 2
            pred_img = backbone_model(input_img, prompt_img)

            label_img2 = F.interpolate(label_img, scale_factor=0.5, mode='bilinear')
            label_img4 = F.interpolate(label_img, scale_factor=0.25, mode='bilinear')

            fal1 = criterion[0](pred_img[0], label_img4)
            fal2 = criterion[0](pred_img[1], label_img2)
            fal3 = criterion[0](pred_img[2], label_img)
            loss_FAL = fal1 + fal2 + fal3

            s1 = criterion[1](pred_img[0], label_img4)
            s2 = criterion[1](pred_img[1], label_img2)
            s3 = criterion[1](pred_img[2], label_img)
            loss_spa = s1 + s2 + s3

            p1 = criterion[2](pred_img[0], label_img4)
            p2 = criterion[2](pred_img[1], label_img2)
            p3 = criterion[2](pred_img[2], label_img)
            loss_percept = p1 + p2 + p3

            loss = loss_FAL + 0.1 * loss_spa + loss_percept

            loss.backward()
            optimizer_backbone.step()

            iter_FAL_adder(loss_FAL.item()),iter_spa_adder(loss_spa.item()), iter_percept_adder(loss_percept.item())
            epoch_FAL_adder(loss_FAL.item()),epoch_spa_adder(loss_spa.item()), epoch_percept_adder(loss_percept.item())

            print(f'\rTime:{iter_timer.toc():.3f}  LR:{scheduler.get_lr()[0]:.10f}  Epoch:{epoch_idx}/{args.num_epochs}  Iter:{iter_idx + 1}/{max_iter} '
                  f'  Loss>> FAL:{iter_FAL_adder.average():.5f}  spa:{iter_spa_adder.average():.5f}  per:{iter_percept_adder.average():.5f}',end='', flush=True)

            writer.add_scalar('FAL Loss', iter_FAL_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)
            writer.add_scalar('spa Loss', iter_spa_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)
            writer.add_scalar('percept Loss', iter_percept_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)

            iter_timer.tic(), iter_FAL_adder.reset(),iter_spa_adder.reset(),iter_percept_adder.reset()

        # one epoch save model
        torch.save({
            'epoch': epoch_idx,
            'max_psnr_1': max_psnr_1,
            'max_ssim_1': max_ssim_1,
            'backbone_model': backbone_model.state_dict(),
            'optimizer_backbone': optimizer_backbone.state_dict(),
        }, os.path.join(args.model_save_dir, 'model.pkl'))

        print(f'\nEPOCH:{epoch_idx}  epoch time:{epoch_timer.toc():.2f}  LR:{scheduler.get_lr()[0]:.10f}  Loss>> FAL:{epoch_FAL_adder.average():.4f}  Spa:{epoch_spa_adder.average():.4f}  Percept:{epoch_percept_adder.average():.4f}',end='', flush=True)
        trainLogger.write(f'\nEPOCH:{epoch_idx}  epoch time:{epoch_timer.toc():.2f}  LR:{scheduler.get_lr()[0]:.10f}  Loss>> FAL:{epoch_FAL_adder.average():.4f}  Spa:{epoch_spa_adder.average():.4f}  Percept:{epoch_percept_adder.average():.4f}')
        epoch_FAL_adder.reset(), epoch_spa_adder.reset(), epoch_percept_adder.reset()
        scheduler.step()

        # one epoch valid
        if epoch_idx % args.valid_epoch == 0:
            avg_psnr_1, avg_ssim_1 = _valid(backbone_model, trainLogger)
            print(f'\nEPOCH:{epoch_idx}  Average PSNR:{avg_psnr_1:.4f}  Average SSIM:{avg_ssim_1:.4f}',end='', flush=True)
            trainLogger.write(f'\nEPOCH:{epoch_idx}   Average PSNR:{avg_psnr_1:.4f}  Average SSIM:{avg_ssim_1:.4f}')

            writer.add_scalar('PSNR', avg_psnr_1, epoch_idx)
            writer.add_scalar('SSIM', avg_ssim_1, epoch_idx)

            if avg_psnr_1 >= max_psnr_1 and avg_ssim_1 >= max_ssim_1:
                max_psnr_1 = max(max_psnr_1, avg_psnr_1)
                best_psnr_epoch_1 = epoch_idx
                max_ssim_1 = avg_ssim_1
                torch.save({
                    'epoch': epoch_idx,
                    'max_psnr_1': max_psnr_1,
                    'max_ssim_1': max_ssim_1,
                    'backbone_model': backbone_model.state_dict(),
                }, os.path.join(args.model_save_dir, 'Best_backbone_psnr_ssim.pkl'))
                print(f'\n===================Best psnr backbone_model saved at epoch:{epoch_idx}  max_psnr_1:{max_psnr_1:.4f}  max_ssim_1:{max_ssim_1:.4f}')
                trainLogger.write(f'\n===================Best psnr backbone_model saved at epoch:{epoch_idx}  max_psnr_1:{max_psnr_1:.4f}  max_ssim_1:{max_ssim_1:.4f}')
            # if avg_ssim_1 >= max_ssim_1:
            #     max_ssim_1 = max(max_ssim_1, avg_ssim_1)
            #     best_ssim_epoch_1 = epoch_idx
            #     torch.save({
            #         'epoch': epoch_idx,
            #         'max_psnr_1': max_psnr_1,
            #         'max_ssim_1': max_ssim_1,
            #         'backbone_model': backbone_model.state_dict(),
            #     }, os.path.join(args.model_save_dir, 'Best_backbone_ssim.pkl'))
            #     print(f'\n===================Best ssim backbone_model saved at epoch:{epoch_idx}  max_psnr_1:{max_psnr_1:.4f}  max_ssim_1:{max_ssim_1:.4f}')
            #     trainLogger.write(f'\n===================Best ssim backbone_model saved at epoch:{epoch_idx}  max_psnr_1:{max_psnr_1:.4f}  max_ssim_1:{max_ssim_1:.4f}')

    print(f'\n')
    trainLogger.write(f'\n')
    print(f'\n max_psnr_1:{max_psnr_1:.4f}  max_ssim_1:{max_ssim_1:.4f}  best_psnr_epoch_1:{best_psnr_epoch_1}  best_ssim_epoch_1:{best_ssim_epoch_1}')
    trainLogger.write(f'\n max_psnr_1:{max_psnr_1:.4f}  max_ssim_1:{max_ssim_1:.4f}  best_psnr_epoch_1:{best_psnr_epoch_1}  best_ssim_epoch_1:{best_ssim_epoch_1}')
    print(f'\ntotal time: {total_time.toc():.2f} hour')
    trainLogger.write(f'\ntotal time: {total_time.toc():.2f} hour')
    writer.close()
    close_log(trainLogger)
