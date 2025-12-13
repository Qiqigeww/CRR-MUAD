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
from dataset import RealIADDataset
import torch.backends.cudnn as cudnn
import argparse
from utils import evaluation_batch_adeval_three, evaluation_batch_adeval_three_visualize
from torch.nn import functional as F
from functools import partial
import warnings
import copy
import logging
from sklearn.metrics import roc_auc_score, average_precision_score
import itertools

warnings.filterwarnings("ignore")

def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'Add_seg_7_3_0.4_log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test(item_list):
    setup_seed(1)

    batch_size = 16
    image_size = 448
    crop_size = 392

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    test_data_list = []
    for i, item in enumerate(item_list):
        root_path = args.data_path

        test_data = RealIADDataset(root=root_path, category=item, transform=data_transform, gt_transform=gt_transform,
                                   phase="test", anomaly_source_path=args.anomaly_source_path, image_size=image_size)
        test_data_list.append(test_data)

    encoder_name = 'dinov2reg_vit_base_14'
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    encoder = vit_encoder.load(encoder_name)

    if 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'large' in encoder_name:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise ValueError("Architecture not in small, base, large.")

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

    model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers,
                   mask_neighbor_size=0, fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder)
    model = model.to(device)

    model.load_state_dict(torch.load('./saved_results/vitill_realiad_uni_dinov2_0.4/model.pth', map_location=device))
    model.eval()

    segmentation_net = DecoderReconstructive(base_width=192, out_channels=2)
    segmentation_net = segmentation_net.to(device)
    segmentation_net.load_state_dict(torch.load('./saved_results/vitill_realiad_uni_dinov2_0.4/Seg_model_7999.pth', map_location=device))
    segmentation_net.eval()
    
    auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
    auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

    for item, test_data in zip(item_list, test_data_list):
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
        results = evaluation_batch_adeval_three_visualize(model, segmentation_net, test_dataloader, device, max_ratio=0.01, resize_mask=256, alpha=0.7)
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

    return

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='/mnt/disk2/wangqs/Datasets/Real-IAD')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str,
                        default='vitill_realiad_uni_dinov2_new2')
    parser.add_argument("--anomaly_source_path", type=str, default='/mnt/disk2/wangqs/Datasets/dtd', help='path to save results')
    args = parser.parse_args()
    item_list = ['audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser', 'fire_hood',
                 'mint', 'mounts', 'pcb', 'phone_battery', 'plastic_nut', 'plastic_plug',
                 'porcelain_doll', 'regulator', 'rolled_strip_base', 'sim_card_set', 'switch', 'tape',
                 'terminalblock', 'toothbrush', 'toy', 'toy_brick', 'transistor1', 'usb',
                 'usb_adaptor', 'u_block', 'vcpill', 'wooden_beads', 'woodstick', 'zipper']
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print_fn(device)

    test(item_list)