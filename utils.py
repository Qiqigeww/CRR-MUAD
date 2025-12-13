import torch
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
from torch.utils.data import DataLoader
from dataset import MVTecDataset
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score, precision_recall_curve, \
    average_precision_score
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from scipy.ndimage import gaussian_filter, binary_dilation
import os
from functools import partial
import math

import pickle

def modify_grad(x, inds, factor=0.):
    inds = inds.expand_as(x)
    x[inds] *= factor
    return x


def modify_grad_v2(x, factor):
    factor = factor.expand_as(x)
    x *= factor
    return x


def global_cosine(a, b, stop_grad=True):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        if stop_grad:
            loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1).detach(),
                                            b[item].view(b[item].shape[0], -1)))
        else:
            loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                            b[item].view(b[item].shape[0], -1)))
    loss = loss / len(a)
    return loss


def global_cosine_hm(a, b, alpha=1., factor=0.):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
        mean_dist = point_dist.mean()
        std_dist = point_dist.reshape(-1).std()

        loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1),
                                        b_.reshape(b_.shape[0], -1)))
        thresh = mean_dist + alpha * std_dist
        partial_func = partial(modify_grad, inds=point_dist < thresh, factor=factor)
        b_.register_hook(partial_func)
    # loss = loss / len(a)
    return loss


##############################################
def cal_discrepancy(fe, fa, OOM, normal, gamma, aggregation=True):
    # normalize the features into uint vector
    
    # from dinomaly_realiad_uni2 import get_logger
    # logger = get_logger('vitill_realiad_uni_dinov2br_c392r_en29_bn4dp4_de8_elaelu_l2g2_i1_it50k_sams2e3_wd1e4_w1hcosa2e4_ghmp09f01w01_b16_s1', './saved_results/vitill_realiad_uni_dinov2br_c392r_en29_bn4dp4_de8_elaelu_l2g2_i1_it50k_sams2e3_wd1e4_w1hcosa2e4_ghmp09f01w01_b16_s1')
    # print_fn = logger.info
    
    fe = F.normalize(fe, p=2, dim=1)
    fa = F.normalize(fa, p=2, dim=1)
    
    if torch.isnan(fe).any() or torch.isnan(fa).any():
        print('In cal_discrepancy fe is {}. fa is {}'.format(fe, fa))
        
    # calculate feature-to-feature discrepancy d_p
    d_p = torch.sum((fe - fa) ** 2, dim=1)
    d_p_no_weight = d_p
    if OOM:
        # if OOM is utilized, we need to calculate the adaptive weights for individual features

        # calculate the mean discrepancy \mu_p to indicate the importance of individual features
        mu_p = torch.mean(d_p)

        if normal:
            # for normal samples: w = ((d_p) / \mu_p)^{\gamma}
            w = (d_p / mu_p) ** gamma

        else:
            # for abnormal samples: w = ((d_p) / \mu_p)^{-\gamma}
            w = (mu_p / d_p) ** gamma

        w = w.detach()

    else:
        # else, we manually assign each feature the same weight, i.e., 1
        w = torch.ones_like(d_p)

    if aggregation:
        d_p = torch.sum(d_p * w)

    sum_w = torch.sum(w)
    
    if torch.isnan(d_p).any():
        print('In cal_discrepancy fe is {}'.format(d_p))
    if torch.isnan(sum_w).any():
        print('In cal_discrepancy sum_w is {}'.format(sum_w))
    
    return d_p, sum_w, d_p_no_weight

def global_cosine_hm_percent(a, b, p=0.9, factor=0.):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
        # mean_dist = point_dist.mean()
        # std_dist = point_dist.reshape(-1).std()
        thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]

        loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1),
                                        b_.reshape(b_.shape[0], -1)))

        partial_func = partial(modify_grad, inds=point_dist < thresh, factor=factor)
        b_.register_hook(partial_func)

    loss = loss / len(a)
    return loss

def regional_cosine_hm_percent(a, b, p=0.9, factor=0.):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
        # mean_dist = point_dist.mean()
        # std_dist = point_dist.reshape(-1).std()
        thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]

        loss += point_dist.mean()

        partial_func = partial(modify_grad, inds=point_dist < thresh, factor=factor)
        b_.register_hook(partial_func)

    loss = loss / len(a)
    return loss


def global_cosine_focal(a, b, p=0.9, alpha=2., min_grad=0.):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1).detach()

        if p < 1.:
            thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]
        else:
            thresh = point_dist.max()
        focal_factor = torch.clip(point_dist, max=thresh) / thresh

        focal_factor = focal_factor ** alpha
        focal_factor = torch.clip(focal_factor, min=min_grad)

        loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1),
                                        b_.reshape(b_.shape[0], -1)))

        partial_func = partial(modify_grad_v2, factor=focal_factor)
        b_.register_hook(partial_func)

    return loss


def regional_cosine_focal(a, b, p=0.9, alpha=2.):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]

        point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
        if p < 1.:
            thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]
        else:
            thresh = point_dist.max()
        focal_factor = torch.clip(point_dist, max=thresh) / thresh
        focal_factor = focal_factor ** alpha

        loss += (point_dist * focal_factor.detach()).mean()

    return loss


def regional_cosine_hm(a, b, p=0.9):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]

        point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
        thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]

        L = point_dist[point_dist >= thresh]
        loss += L.mean()

    return loss


def region_cosine(a, b, stop_grad=True):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += 1 - cos_loss(a[item].detach(), b[item]).mean()
    return loss


def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='add', norm_factor=None):
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)
    if amap_mode == 'mul':
        anomaly_map = np.ones(out_size)
    else:
        anomaly_map = np.zeros(out_size)

    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        if norm_factor is not None:
            a_map = 0.1 * (a_map - norm_factor[0][i]) / (norm_factor[1][i] - norm_factor[0][i])

        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list


def cal_anomaly_maps(fs_list, ft_list, out_size=224):
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)

    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map_list.append(a_map)
    anomaly_map = torch.cat(a_map_list, dim=1).mean(dim=1, keepdim=True)
    return anomaly_map, a_map_list


def map_normalization(fs_list, ft_list, start=0.5, end=0.95):
    start_list = []
    end_list = []
    with torch.no_grad():
        for i in range(len(ft_list)):
            fs = fs_list[i]
            ft = ft_list[i]
            a_map = 1 - F.cosine_similarity(fs, ft)
            start_list.append(torch.quantile(a_map, q=start).item())
            end_list.append(torch.quantile(a_map, q=end).item())

    return [start_list, end_list]


def cal_anomaly_map_v2(fs_list, ft_list, out_size=224, amap_mode='add'):
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size // 4, mode='bilinear', align_corners=False)
        a_map_list.append(a_map)

    anomaly_map = torch.stack(a_map_list, dim=-1).sum(-1)
    anomaly_map = F.interpolate(anomaly_map, size=out_size, mode='bilinear', align_corners=False)
    anomaly_map = anomaly_map[0, 0, :, :].to('cpu').detach().numpy()

    return anomaly_map, a_map_list


def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map) / 255 + np.float32(img) / 255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)


def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


def return_best_thr(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    thrs = thrs[~np.isnan(f1s)]
    f1s = f1s[~np.isnan(f1s)]
    best_thr = thrs[np.argmax(f1s)]
    return best_thr


def f1_score_max(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    return f1s.max()


def specificity_score(y_true, y_score):
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    TN = (y_true[y_score == 0] == 0).sum()
    N = (y_true == 0).sum()
    return TN / N


def evaluation(model, dataloader, device, _class_=None, calc_pro=True, norm_factor=None, feature_used='all',
               max_ratio=0):
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []

    with torch.no_grad():
        for img, gt, label, _ in dataloader:
            img = img.to(device)

            en, de = model(img)

            if feature_used == 'trained':
                anomaly_map, _ = cal_anomaly_map(en[3:], de[3:], img.shape[-1], amap_mode='a', norm_factor=norm_factor)
            elif feature_used == 'freezed':
                anomaly_map, _ = cal_anomaly_map(en[:3], de[:3], img.shape[-1], amap_mode='a', norm_factor=norm_factor)
            else:
                anomaly_map, _ = cal_anomaly_map(en, de, img.shape[-1], amap_mode='a', norm_factor=norm_factor)
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            # gt[gt > 0.5] = 1
            # gt[gt <= 0.5] = 0
            gt = gt.bool()

            if calc_pro:
                if label.item() != 0:
                    aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
                                                  anomaly_map[np.newaxis, :, :]))
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            if max_ratio <= 0:
                sp_score = anomaly_map.max()
            else:
                anomaly_map = anomaly_map.ravel()
                sp_score = np.sort(anomaly_map)[-int(anomaly_map.shape[0] * max_ratio):]
                sp_score = sp_score.mean()
            pr_list_sp.append(sp_score)
        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 4)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)

    return auroc_px, auroc_sp, round(np.mean(aupro_list), 4)


##############################################
def evaluation_batch_adeval(model, dataloader, device, _class_=None, max_ratio=0, resize_mask=None, use_adeval=True):
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for img, gt, label, img_path in dataloader:
            img = img.to(device)
            # starter.record()
            output = model(img)
            # ender.record()
            # torch.cuda.synchronize()
            # curr_time = starter.elapsed_time(ender)
            en, de = output[0], output[1]

            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            # anomaly_map = anomaly_map - anomaly_map.mean(dim=[1, 2, 3]).view(-1, 1, 1, 1)

            if resize_mask is not None:
                anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear', align_corners=False)
                gt = F.interpolate(gt, size=resize_mask, mode='bilinear')
                gt = torch.where(gt < 0.5, torch.zeros_like(gt), torch.ones_like(gt))

            anomaly_map = gaussian_kernel(anomaly_map)

            gt = gt.bool()
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]

            gt_list_px.append(gt)
            pr_list_px.append(anomaly_map)
            gt_list_sp.append(label)

            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
            else:
                anomaly_map = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1)
            pr_list_sp.append(sp_score)

        gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
        pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
        gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
        pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()

        if use_adeval:
            from adeval import EvalAccumulatorCuda
            score_min = min(pr_list_sp) - 1e-7
            score_max = max(pr_list_sp) + 1e-7
            anomap_min = pr_list_px.min()
            anomap_max = pr_list_px.max()

            accum = EvalAccumulatorCuda(score_min, score_max, anomap_min, anomap_max)

            accum.add_anomap_batch(torch.tensor(pr_list_px).cuda(non_blocking=True),
                                   torch.tensor(gt_list_px.astype(np.uint8)).cuda(non_blocking=True))

            # for i in range(len(pr_list_sp)):
            #     accum.add_image(torch.tensor(pr_list_sp[i]), torch.tensor(gt_list_sp[i]))

            metrics = accum.summary()   
        
        auroc_sp = roc_auc_score(gt_list_sp, pr_list_sp)
        ap_sp = average_precision_score(gt_list_sp, pr_list_sp)
        f1_sp = f1_score_max(gt_list_sp, pr_list_sp)
        f1_px = f1_score_max(gt_list_px.ravel(), pr_list_px.ravel())   
          
        if use_adeval:
            auroc_px = metrics['p_auroc']
            ap_px = metrics['p_aupr']
            aupro_px = metrics['p_aupro']
        else:
            auroc_px = roc_auc_score(gt_list_px.ravel(), pr_list_px.ravel())
            ap_px = average_precision_score(gt_list_px.ravel(), pr_list_px.ravel())
            aupro_px = compute_pro(gt_list_px, pr_list_px)

    return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px]


def evaluation_batch_adeval_new(model, segmentation_net, dataloader, device, _class_=None, max_ratio=0, resize_mask=None, use_adeval=True):
    segmentation_net.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for img, gt, label, img_path in dataloader:
            img = img.to(device)
            # starter.record()
            output = model(img)
            # ender.record()
            # torch.cuda.synchronize()
            # curr_time = starter.elapsed_time(ender)
            en, de, seg_input = output[0], output[1], output[2]
            output_segmentation = segmentation_net(seg_input)
            output_segmentation = torch.softmax(output_segmentation, dim=1)[:, 1:, :, :]
            anomaly_map_ori, _ = cal_anomaly_maps(en, de, resize_mask)

            if resize_mask is not None:
                if output_segmentation.shape[-1] != resize_mask:
                    anomaly_map_seg = F.interpolate(output_segmentation, size=resize_mask, mode='bilinear', align_corners=False)
                else:
                    anomaly_map_seg = output_segmentation
                anomaly_map = anomaly_map_ori + anomaly_map_seg
                
                gt = F.interpolate(gt, size=resize_mask, mode='bilinear')
                gt = torch.where(gt < 0.5, torch.zeros_like(gt), torch.ones_like(gt))

            anomaly_map = gaussian_kernel(anomaly_map)

            gt = gt.bool()
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]

            gt_list_px.append(gt)
            pr_list_px.append(anomaly_map)
            gt_list_sp.append(label)

            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
            else:
                anomaly_map = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1)
            pr_list_sp.append(sp_score)

        gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
        pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
        gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
        pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()

        if use_adeval:
            from adeval import EvalAccumulatorCuda
            score_min = min(pr_list_sp) - 1e-7
            score_max = max(pr_list_sp) + 1e-7
            anomap_min = pr_list_px.min()
            anomap_max = pr_list_px.max()

            accum = EvalAccumulatorCuda(score_min, score_max, anomap_min, anomap_max)

            accum.add_anomap_batch(torch.tensor(pr_list_px).cuda(non_blocking=True),
                                   torch.tensor(gt_list_px.astype(np.uint8)).cuda(non_blocking=True))

            # for i in range(len(pr_list_sp)):
            #     accum.add_image(torch.tensor(pr_list_sp[i]), torch.tensor(gt_list_sp[i]))

            metrics = accum.summary()   
        
        auroc_sp = roc_auc_score(gt_list_sp, pr_list_sp)
        ap_sp = average_precision_score(gt_list_sp, pr_list_sp)
        f1_sp = f1_score_max(gt_list_sp, pr_list_sp)
        f1_px = f1_score_max(gt_list_px.ravel(), pr_list_px.ravel())   
          
        if use_adeval:
            auroc_px = metrics['p_auroc']
            ap_px = metrics['p_aupr']
            aupro_px = metrics['p_aupro']
        else:
            auroc_px = roc_auc_score(gt_list_px.ravel(), pr_list_px.ravel())
            ap_px = average_precision_score(gt_list_px.ravel(), pr_list_px.ravel())
            aupro_px = compute_pro(gt_list_px, pr_list_px)

    return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px]


def evaluation_batch_adeval_three(model, segmentation_net, dataloader, device, _class_=None, max_ratio=0, resize_mask=None, use_adeval=True, fusion_method='weighted_sum', alpha=0.5):
    segmentation_net.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # display_gt_images = torch.zeros((16 ,3 ,256 ,256)).cuda()
    # display_out_masks = torch.zeros((16 ,1 ,256 ,256)).cuda()
    # display_in_masks = torch.zeros((16 ,1 ,256 ,256)).cuda()
    # cnt_display = 0
    # display_indices = np.random.randint(len(dataloader), size=(16,))

    with torch.no_grad():
        # for img, gt, label, img_path in dataloader:
        for i_batch, (img, gt, label, img_path) in enumerate(dataloader):
            img = img.to(device)
            # starter.record()
            output = model(img)
            # ender.record()
            # torch.cuda.synchronize()
            # curr_time = starter.elapsed_time(ender)
            en, de, seg_input = output[0], output[1], output[2]
            output_segmentation = segmentation_net(seg_input)
            output_segmentation = torch.softmax(output_segmentation, dim=1)[:, 1:, :, :]
            anomaly_map_ori, _ = cal_anomaly_maps(en, de, resize_mask)

            if resize_mask is not None:
                if output_segmentation.shape[-1] != resize_mask:
                    anomaly_map_seg = F.interpolate(output_segmentation, size=resize_mask, mode='bilinear', align_corners=False)
                else:
                    anomaly_map_seg = output_segmentation
                gt = F.interpolate(gt, size=resize_mask, mode='bilinear')
                gt = torch.where(gt < 0.5, torch.zeros_like(gt), torch.ones_like(gt))

            # 融合方法
            if fusion_method == 'weighted_sum':
                anomaly_map = alpha * anomaly_map_ori + (1 - alpha) * anomaly_map_seg
            elif fusion_method == 'max':
                anomaly_map = torch.max(anomaly_map_ori, anomaly_map_seg)
            elif fusion_method == 'min':
                anomaly_map = torch.min(anomaly_map_ori, anomaly_map_seg)
            else:
                raise ValueError(f"Unknown fusion method: {fusion_method}")

            anomaly_map = gaussian_kernel(anomaly_map)

            # if i_batch in display_indices:
            #     display_gt_images[cnt_display] = img[0]
            #     display_out_masks[cnt_display] = anomaly_map[0]
            #     display_in_masks[cnt_display] = gt[0]
            #     cnt_display += 1

            gt = gt.bool()
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]

            gt_list_px.append(gt)
            pr_list_px.append(anomaly_map)
            gt_list_sp.append(label)

            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
            else:
                anomaly_map = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1)
            pr_list_sp.append(sp_score)

        gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
        pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
        gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
        pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()

        if use_adeval:
            from adeval import EvalAccumulatorCuda
            score_min = min(pr_list_sp) - 1e-7
            score_max = max(pr_list_sp) + 1e-7
            anomap_min = pr_list_px.min()
            anomap_max = pr_list_px.max()

            accum = EvalAccumulatorCuda(score_min, score_max, anomap_min, anomap_max)

            accum.add_anomap_batch(torch.tensor(pr_list_px).cuda(non_blocking=True),
                                   torch.tensor(gt_list_px.astype(np.uint8)).cuda(non_blocking=True))

            # for i in range(len(pr_list_sp)):
            #     accum.add_image(torch.tensor(pr_list_sp[i]), torch.tensor(gt_list_sp[i]))

            metrics = accum.summary()   
        
        auroc_sp = roc_auc_score(gt_list_sp, pr_list_sp)
        ap_sp = average_precision_score(gt_list_sp, pr_list_sp)
        f1_sp = f1_score_max(gt_list_sp, pr_list_sp)
        f1_px = f1_score_max(gt_list_px.ravel(), pr_list_px.ravel())   
          
        if use_adeval:
            auroc_px = metrics['p_auroc']
            ap_px = metrics['p_aupr']
            aupro_px = metrics['p_aupro']
        else:
            auroc_px = roc_auc_score(gt_list_px.ravel(), pr_list_px.ravel())
            ap_px = average_precision_score(gt_list_px.ravel(), pr_list_px.ravel())
            aupro_px = compute_pro(gt_list_px, pr_list_px)

    return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px]

# def visualize(model, dataloader, device, _class_='None', save_name='save'):
#     model.eval()
#     save_dir = os.path.join('./visualize', save_name)
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

#     with torch.no_grad():
#         for img, gt, label, img_path in dataloader:
#             img = img.to(device)
#             output = model(img)
#             en, de = output[0], output[1]
#             anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
#             anomaly_map = gaussian_kernel(anomaly_map)

#             for i in range(0, anomaly_map.shape[0], 8):
#                 heatmap = min_max_norm(anomaly_map[i, 0].cpu().numpy())
#                 heatmap = cvt2heatmap(heatmap * 255)
#                 im = img[i].permute(1, 2, 0).cpu().numpy()
#                 im = im * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
#                 im = (im * 255).astype('uint8')
#                 im = im[:, :, ::-1]
#                 hm_on_img = show_cam_on_image(im, heatmap)
#                 mask = (gt[i][0].numpy() * 255).astype('uint8')
#                 save_dir_class = os.path.join(save_dir, str(_class_))
#                 if not os.path.exists(save_dir_class):
#                     os.mkdir(save_dir_class)
#                 name = img_path[i].split('/')[-2] + '_' + img_path[i].split('/')[-1].replace('.png', '')
#                 cv2.imwrite(save_dir_class + '/' + name + '_img.png', im)
#                 cv2.imwrite(save_dir_class + '/' + name + '_cam.png', hm_on_img)
#                 cv2.imwrite(save_dir_class + '/' + name + '_gt.png', mask)

#     return

def evaluation_batch_adeval_three_visualize(model, segmentation_net, dataloader, device, max_ratio=0, resize_mask=None, use_adeval=True, fusion_method='weighted_sum', alpha=0.5, save_name='save', _class_='None'):
    segmentation_net.eval()
    save_dir = os.path.join('./visualize', save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)    
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    with torch.no_grad():
        for img, gt, label, img_path in dataloader:
            img = img.to(device)
            output = model(img)
            en, de, seg_input = output[0], output[1], output[2]
            output_segmentation = segmentation_net(seg_input)
            output_segmentation = torch.softmax(output_segmentation, dim=1)[:, 1:, :, :]
            anomaly_map_ori, _ = cal_anomaly_maps(en, de, resize_mask)

            if resize_mask is not None:
                if output_segmentation.shape[-1] != resize_mask:
                    anomaly_map_seg = F.interpolate(output_segmentation, size=resize_mask, mode='bilinear', align_corners=False)
                else:
                    anomaly_map_seg = output_segmentation
                gt = F.interpolate(gt, size=resize_mask, mode='bilinear')
                gt = torch.where(gt < 0.5, torch.zeros_like(gt), torch.ones_like(gt))
                
                img = F.interpolate(img, size=resize_mask, mode='bilinear', align_corners=False)

            # 融合方法
            if fusion_method == 'weighted_sum':
                anomaly_map = alpha * anomaly_map_ori + (1 - alpha) * anomaly_map_seg
            elif fusion_method == 'max':
                anomaly_map = torch.max(anomaly_map_ori, anomaly_map_seg)
            elif fusion_method == 'min':
                anomaly_map = torch.min(anomaly_map_ori, anomaly_map_seg)
            else:
                raise ValueError(f"Unknown fusion method: {fusion_method}")

            anomaly_map = gaussian_kernel(anomaly_map)

            for i in range(0, anomaly_map.shape[0], 8):
                heatmap = min_max_norm(anomaly_map[i, 0].cpu().numpy())
                heatmap = cvt2heatmap(heatmap * 255)
                im = img[i].permute(1, 2, 0).cpu().numpy()
                im = im * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                im = (im * 255).astype('uint8')
                im = im[:, :, ::-1]
                hm_on_img = show_cam_on_image(im, heatmap)
                mask = (gt[i][0].numpy() * 255).astype('uint8')

                name = img_path[i].split('/')[-2] + '_' + img_path[i].split('/')[-1].replace('.png', '')
                cv2.imwrite(save_dir + '/' + name + '_img.png', im)
                cv2.imwrite(save_dir + '/' + name + '_cam.png', hm_on_img)
                cv2.imwrite(save_dir + '/' + name + '_gt.png', mask)            

            gt = gt.bool()
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]

            gt_list_px.append(gt)
            pr_list_px.append(anomaly_map)
            gt_list_sp.append(label)

            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
            else:
                anomaly_map = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1)
            pr_list_sp.append(sp_score)

        gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
        pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
        gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
        pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()

        if use_adeval:
            from adeval import EvalAccumulatorCuda
            score_min = min(pr_list_sp) - 1e-7
            score_max = max(pr_list_sp) + 1e-7
            anomap_min = pr_list_px.min()
            anomap_max = pr_list_px.max()

            accum = EvalAccumulatorCuda(score_min, score_max, anomap_min, anomap_max)

            accum.add_anomap_batch(torch.tensor(pr_list_px).cuda(non_blocking=True),
                                   torch.tensor(gt_list_px.astype(np.uint8)).cuda(non_blocking=True))

            # for i in range(len(pr_list_sp)):
            #     accum.add_image(torch.tensor(pr_list_sp[i]), torch.tensor(gt_list_sp[i]))

            metrics = accum.summary()   
        
        auroc_sp = roc_auc_score(gt_list_sp, pr_list_sp)
        ap_sp = average_precision_score(gt_list_sp, pr_list_sp)
        f1_sp = f1_score_max(gt_list_sp, pr_list_sp)
        f1_px = f1_score_max(gt_list_px.ravel(), pr_list_px.ravel())   
          
        if use_adeval:
            auroc_px = metrics['p_auroc']
            ap_px = metrics['p_aupr']
            aupro_px = metrics['p_aupro']
        else:
            auroc_px = roc_auc_score(gt_list_px.ravel(), pr_list_px.ravel())
            ap_px = average_precision_score(gt_list_px.ravel(), pr_list_px.ravel())
            aupro_px = compute_pro(gt_list_px, pr_list_px)

    return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px]

##############################################


def evaluation_batch(model, dataloader, device, _class_=None, max_ratio=0, resize_mask=None):
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for img, gt, label, img_path in dataloader:
            img = img.to(device)
            # starter.record()
            output = model(img)
            # ender.record()
            # torch.cuda.synchronize()
            # curr_time = starter.elapsed_time(ender)
            en, de = output[0], output[1]

            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            # anomaly_map = anomaly_map - anomaly_map.mean(dim=[1, 2, 3]).view(-1, 1, 1, 1)

            if resize_mask is not None:
                anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear', align_corners=False)
                gt = F.interpolate(gt, size=resize_mask, mode='nearest')

            anomaly_map = gaussian_kernel(anomaly_map)

            gt = gt.bool()
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]

            gt_list_px.append(gt)
            pr_list_px.append(anomaly_map)
            gt_list_sp.append(label)

            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
            else:
                anomaly_map = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1)
            pr_list_sp.append(sp_score)

        gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
        pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
        gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
        pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()

        aupro_px = compute_pro(gt_list_px, pr_list_px)

        gt_list_px, pr_list_px = gt_list_px.ravel(), pr_list_px.ravel()

        auroc_px = roc_auc_score(gt_list_px, pr_list_px)
        auroc_sp = roc_auc_score(gt_list_sp, pr_list_sp)
        ap_px = average_precision_score(gt_list_px, pr_list_px)
        ap_sp = average_precision_score(gt_list_sp, pr_list_sp)

        f1_sp = f1_score_max(gt_list_sp, pr_list_sp)
        f1_px = f1_score_max(gt_list_px, pr_list_px)

    return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px]


def evaluation_batch_loco(model, dataloader, device, _class_=None, max_ratio=0):
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    defect_type_list = []
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    with torch.no_grad():
        for img, gt, label, path, defect_type, size in dataloader:
            img = img.to(device)

            output = model(img)
            en, de = output[0], output[1]

            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            anomaly_map = gaussian_kernel(anomaly_map)

            gt = gt.bool()

            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.cpu().numpy().ravel())
            gt_list_sp.extend(label.cpu().numpy().astype(int))

            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0].cpu().numpy()
            else:
                anomaly_map = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1).cpu().numpy()
            pr_list_sp.extend(sp_score)
            defect_type_list.extend(defect_type)

        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 4)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)
        ap_px = round(average_precision_score(gt_list_px, pr_list_px), 4)
        ap_sp = round(average_precision_score(gt_list_sp, pr_list_sp), 4)

        defect_type_list = np.array(defect_type_list)
        auroc_logic = roc_auc_score(
            np.array(gt_list_sp)[np.logical_or(defect_type_list == 'good', defect_type_list == 'logical_anomalies')],
            np.array(pr_list_sp)[np.logical_or(defect_type_list == 'good', defect_type_list == 'logical_anomalies')])
        auroc_struct = roc_auc_score(
            np.array(gt_list_sp)[np.logical_or(defect_type_list == 'good', defect_type_list == 'structural_anomalies')],
            np.array(pr_list_sp)[np.logical_or(defect_type_list == 'good', defect_type_list == 'structural_anomalies')])
        auroc_both = (auroc_logic + auroc_struct) / 2

    return auroc_sp, auroc_logic, auroc_struct, auroc_both


def evaluation_uniad(model, dataloader, device, _class_=None, reg_calib=False, max_ratio=0):
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    with torch.no_grad():
        for img, gt, label, _ in dataloader:
            img = img.to(device)
            if reg_calib:
                en, de, reg = model({'image': img})
            else:
                en, de = model({'image': img})

            anomaly_map = torch.mean(F.mse_loss(de, en, reduction='none'), dim=1, keepdim=True)
            anomaly_map = F.interpolate(anomaly_map, size=(img.shape[-1], img.shape[-1]), mode='bilinear',
                                        align_corners=False)

            if reg_calib:
                if reg.shape[1] == 2:
                    reg_mean = reg[:, 0].view(-1, 1, 1, 1)
                    reg_max = reg[:, 1].view(-1, 1, 1, 1)
                    anomaly_map = (anomaly_map - reg_mean) / (reg_max - reg_mean)
                    # anomaly_map = anomaly_map - reg_max

                else:
                    reg = F.interpolate(reg, size=img.shape[-1], mode='bilinear', align_corners=True)
                    anomaly_map = anomaly_map - reg

            anomaly_map = gaussian_kernel(anomaly_map)

            gt = gt.bool()

            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.cpu().numpy().ravel())
            gt_list_sp.extend(label.cpu().numpy().astype(int))

            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0].cpu().numpy()
            else:
                anomaly_map = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1).cpu().numpy()
            pr_list_sp.extend(sp_score)

        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 4)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)
        ap_px = round(average_precision_score(gt_list_px, pr_list_px), 4)
        ap_sp = round(average_precision_score(gt_list_sp, pr_list_sp), 4)

    return auroc_px, auroc_sp, ap_px, ap_sp, [gt_list_px, pr_list_px, gt_list_sp, pr_list_sp]


def visualize(model, dataloader, device, _class_='None', save_name='save'):
    model.eval()
    save_dir = os.path.join('./visualize', save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    with torch.no_grad():
        for img, gt, label, img_path in dataloader:
            img = img.to(device)
            output = model(img)
            en, de = output[0], output[1]
            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            anomaly_map = gaussian_kernel(anomaly_map)

            for i in range(0, anomaly_map.shape[0], 8):
                heatmap = min_max_norm(anomaly_map[i, 0].cpu().numpy())
                heatmap = cvt2heatmap(heatmap * 255)
                im = img[i].permute(1, 2, 0).cpu().numpy()
                im = im * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                im = (im * 255).astype('uint8')
                im = im[:, :, ::-1]
                hm_on_img = show_cam_on_image(im, heatmap)
                mask = (gt[i][0].numpy() * 255).astype('uint8')
                save_dir_class = os.path.join(save_dir, str(_class_))
                if not os.path.exists(save_dir_class):
                    os.mkdir(save_dir_class)
                name = img_path[i].split('/')[-2] + '_' + img_path[i].split('/')[-1].replace('.png', '')
                cv2.imwrite(save_dir_class + '/' + name + '_img.png', im)
                cv2.imwrite(save_dir_class + '/' + name + '_cam.png', hm_on_img)
                cv2.imwrite(save_dir_class + '/' + name + '_gt.png', mask)

    return


def save_feature(model, dataloader, device, _class_='None', save_name='save'):
    model.eval()
    save_dir = os.path.join('./feature', save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with torch.no_grad():
        for img, gt, label, img_path in dataloader:
            img = img.to(device)
            en, de = model(img)

            en_abnorm_list = []
            en_normal_list = []
            de_abnorm_list = []
            de_normal_list = []

            for i in range(3):
                en_feat = en[0 + i]
                de_feat = de[0 + i]

                gt_resize = F.interpolate(gt, size=en_feat.shape[2], mode='bilinear') > 0

                en_abnorm = en_feat.permute(0, 2, 3, 1)[gt_resize.permute(0, 2, 3, 1)[:, :, :, 0]]
                en_normal = en_feat.permute(0, 2, 3, 1)[gt_resize.permute(0, 2, 3, 1)[:, :, :, 0] == 0]

                de_abnorm = de_feat.permute(0, 2, 3, 1)[gt_resize.permute(0, 2, 3, 1)[:, :, :, 0]]
                de_normal = de_feat.permute(0, 2, 3, 1)[gt_resize.permute(0, 2, 3, 1)[:, :, :, 0] == 0]

                en_abnorm_list.append(F.normalize(en_abnorm, dim=1).cpu().numpy())
                en_normal_list.append(F.normalize(en_normal, dim=1).cpu().numpy())
                de_abnorm_list.append(F.normalize(de_abnorm, dim=1).cpu().numpy())
                de_normal_list.append(F.normalize(de_normal, dim=1).cpu().numpy())

            save_dir_class = os.path.join(save_dir, str(_class_))
            if not os.path.exists(save_dir_class):
                os.mkdir(save_dir_class)
            name = img_path[0].split('/')[-2] + '_' + img_path[0].split('/')[-1].replace('.png', '')

            saved_dict = {'en_abnorm_list': en_abnorm_list, 'en_normal_list': en_normal_list,
                          'de_abnorm_list': de_abnorm_list, 'de_normal_list': de_normal_list}

            with open(save_dir_class + '/' + name + '.pkl', 'wb') as f:
                pickle.dump(saved_dict, f)

    return


def visualize_noseg(model, dataloader, device, _class_='None', save_name='save'):
    model.eval()
    save_dir = os.path.join('./visualize', save_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with torch.no_grad():
        for img, label, img_path in dataloader:
            img = img.to(device)
            en, de = model(img)

            anomaly_map, _ = cal_anomaly_map(en, de, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            heatmap = min_max_norm(anomaly_map)
            heatmap = cvt2heatmap(heatmap * 255)
            img = img.permute(0, 2, 3, 1).cpu().numpy()[0]
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = (img * 255).astype('uint8')
            hm_on_img = show_cam_on_image(img, heatmap)

            save_dir_class = os.path.join(save_dir, str(_class_))
            if not os.path.exists(save_dir_class):
                os.mkdir(save_dir_class)
            name = img_path[0].split('/')[-2] + '_' + img_path[0].split('/')[-1].replace('.png', '')
            cv2.imwrite(save_dir_class + '/' + name + '_seg.png', heatmap)
            cv2.imwrite(save_dir_class + '/' + name + '_cam.png', hm_on_img)

    return


def visualize_loco(model, dataloader, device, _class_='None', save_name='save'):
    model.eval()
    save_dir = os.path.join('./visualize', save_name)
    with torch.no_grad():
        for img, gt, label, img_path, defect_type, size in dataloader:
            img = img.to(device)
            en, de = model(img)

            anomaly_map, _ = cal_anomaly_map(en, de, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            anomaly_map = cv2.resize(anomaly_map, dsize=(size[0].item(), size[1].item()),
                                     interpolation=cv2.INTER_NEAREST)

            save_dir_class = os.path.join(save_dir, str(_class_), 'test', defect_type[0])
            if not os.path.exists(save_dir_class):
                os.makedirs(save_dir_class)
            name = img_path[0].split('/')[-1].replace('.png', '')
            cv2.imwrite(save_dir_class + '/' + name + '.tiff', anomaly_map)
    return


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:
    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                      groups=channels,
                                      bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


class FeatureJitter(torch.nn.Module):
    def __init__(self, scale=1., p=0.25) -> None:
        super(FeatureJitter, self).__init__()
        self.scale = scale
        self.p = p

    def add_jitter(self, feature):
        if self.scale > 0:
            B, C, H, W = feature.shape
            feature_norms = feature.norm(dim=1).unsqueeze(1) / C  # B*1*H*W
            jitter = torch.randn((B, C, H, W), device=feature.device)
            jitter = F.normalize(jitter, dim=1)
            jitter = jitter * feature_norms * self.scale
            mask = torch.rand((B, 1, H, W), device=feature.device) < self.p
            feature = feature + jitter * mask
        return feature

    def forward(self, x):
        if self.training:
            x = self.add_jitter(x)
        return x


def replace_layers(model, old, new):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers(module, old, new)

        if isinstance(module, old):
            ## simple module
            setattr(model, n, new)


from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class WarmCosineScheduler(_LRScheduler):

    def __init__(self, optimizer, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, ):
        self.final_value = final_value
        self.total_iters = total_iters
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((warmup_schedule, schedule))

        super(WarmCosineScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.total_iters:
            return [self.final_value for base_lr in self.base_lrs]
        else:
            return [self.schedule[self.last_epoch] for base_lr in self.base_lrs]

####################################################################################
import torch.nn as nn
class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        return loss