# This script is designed to perform the second-stage training of the segmentation model.

import torch
import torch.nn as nn
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader, ConcatDataset

from models.uad import ViTill
from models import vit_encoder
from torch.nn.init import trunc_normal_
from models.vision_transformer import Block as VitBlock, bMlp6, LinearAttention2, DecoderReconstructive
from dataset import MVTecDataset, RealIADDataset
import torch.backends.cudnn as cudnn
import argparse
from utils import evaluation_batch_adeval_new, FocalLoss
from torch.nn import functional as F
from functools import partial
from ptflops import get_model_complexity_info
from optimizers import StableAdamW
import warnings
import copy
import logging
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'Add_seg_log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train(item_list):
    setup_seed(1)

    total_iters = 8000 #50000
    # seg_start_iters = 50000
    batch_size = 8
    image_size = 448
    crop_size = 392

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    train_data_list = []
    test_data_list = []
    for i, item in enumerate(item_list):
        # root_path = '/data/disk8T2/guoj/Real-IAD'
        root_path = args.data_path

        train_data = RealIADDataset(root=root_path, category=item, transform=data_transform, gt_transform=gt_transform,
                                    phase='train', anomaly_source_path=args.anomaly_source_path, image_size=image_size)
        train_data.classes = item
        train_data.class_to_idx = {item: i}
        # train_data.samples = [(sample[0], i) for sample in train_data.samples]

        test_data = RealIADDataset(root=root_path, category=item, transform=data_transform, gt_transform=gt_transform,
                                   phase="test", anomaly_source_path=args.anomaly_source_path, image_size=image_size)
        train_data_list.append(train_data)
        test_data_list.append(test_data)

    train_data = ConcatDataset(train_data_list)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   drop_last=True)

    # encoder_name = 'dinov2reg_vit_small_14'
    encoder_name = 'dinov2reg_vit_base_14'
    # encoder_name = 'dinov2reg_vit_large_14'

    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    # target_layers = list(range(4, 19))

    encoder = vit_encoder.load(encoder_name)

    if 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'large' in encoder_name:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise "Architecture not in small, base, large."

    bottleneck = []
    decoder = []

    bottleneck.append(bMlp6(embed_dim, embed_dim * 4, embed_dim, drop=0.4, lambda_mgd=0.4))
    bottleneck = nn.ModuleList(bottleneck)

    for i in range(8):
        blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8), attn_drop=0.,
                       attn=LinearAttention2)
        decoder.append(blk)
    decoder = nn.ModuleList(decoder)

    ##############################################
    segmentation_net = DecoderReconstructive(base_width=192, out_channels=2)
    ##############################################

    model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers,
                   mask_neighbor_size=0, fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder)
    model = model.to(device)
    ###############################################
    segmentation_net = segmentation_net.to(device)
    segmentation_net.apply(weights_init)
    model.load_state_dict(torch.load('./saved_results/vitill_realiad_uni_dinov2_0.4/model.pth', map_location=device))
    model.eval()    

    ##############################################
    loss_focal = FocalLoss()
    seg_optimizer = torch.optim.Adam([{"params": segmentation_net.parameters(), "lr": 0.0001}])
    seg_scheduler = torch.optim.lr_scheduler.MultiStepLR(seg_optimizer,[15680,17640],gamma=0.2, last_epoch=-1)    
    ##############################################
    print_fn('train image number:{}'.format(len(train_data)))

    it = 0
    for epoch in range(int(np.ceil(total_iters / len(train_dataloader)))):
        segmentation_net.train()

        loss_list = []     
        for (img, aug_img, aug_mask) in train_dataloader:
            seg_optimizer.zero_grad()
            img = img.to(device)
            aug_img = aug_img.to(device)
            mask = aug_mask.to(device)

            # en, de, _ = model(img)
            ##############################################
            with torch.no_grad():
                en_aug, de_aug, output = model(aug_img)
                
            output_segmentation = segmentation_net(output)
            output_segmentation = torch.softmax(output_segmentation, dim=1)
            ##############################################

            ##############################################
            mask = F.interpolate(
                mask,
                size=output_segmentation.size()[2:],
                mode="bilinear",
                align_corners=False,
            )
            mask = torch.where(
                mask < 0.5, torch.zeros_like(mask), torch.ones_like(mask)
            )
            
            loss = loss_focal(output_segmentation, mask)            
            ##############################################
         
            loss.backward()
            # nn.utils.clip_grad_norm(segmentation_net.parameters(), max_norm=0.1)

            seg_optimizer.step()
            loss_list.append(loss.item())                                      

            if (it + 1) % 4000 == 0:#
                torch.save(segmentation_net.state_dict(), os.path.join(args.save_dir, args.save_name, f'Seg_model_{it}.pth'))

                auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
                auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

                for item, test_data in zip(item_list, test_data_list):
                    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                                                  num_workers=4)
                    results = evaluation_batch_adeval_new(model, segmentation_net, test_dataloader, device, max_ratio=0.01, resize_mask=256)
                    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results

                    auroc_sp_list.append(auroc_sp)
                    ap_sp_list.append(ap_sp)
                    f1_sp_list.append(f1_sp)
                    auroc_px_list.append(auroc_px)
                    ap_px_list.append(ap_px)
                    f1_px_list.append(f1_px)
                    aupro_px_list.append(aupro_px)

                    print_fn(
                        '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                            item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))

                print_fn(
                    'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                        np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                        np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))

                segmentation_net.train()    

            it += 1
            if it == total_iters:
                break
            
            if (it + 1) % 100 == 0:
                print_fn(
                    'iter [{}/{}], loss: {:.4f}'.format(
                        it, total_iters, np.mean(loss_list)
                    )
                )
                loss_list = []
            seg_scheduler.step()
    torch.save(segmentation_net.state_dict(), os.path.join(args.save_dir, args.save_name, 'Seg_model_last.pth'))

    return


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "2"
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='/mnt/disk2/xxx/Datasets/Real-IAD')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str,
                        default='vitill_realiad_uni_dinov2_0.4')
    parser.add_argument("--anomaly_source_path", type=str, default='/mnt/disk2/xxx/Datasets/dtd', help='path to save results')
    args = parser.parse_args()
    item_list = ['audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser', 'fire_hood',
                 'mint', 'mounts', 'pcb', 'phone_battery', 'plastic_nut', 'plastic_plug',
                 'porcelain_doll', 'regulator', 'rolled_strip_base', 'sim_card_set', 'switch', 'tape',
                 'terminalblock', 'toothbrush', 'toy', 'toy_brick', 'transistor1', 'usb',
                 'usb_adaptor', 'u_block', 'vcpill', 'wooden_beads', 'woodstick', 'zipper']
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    print_fn(device)

    train(item_list)