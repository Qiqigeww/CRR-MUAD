<div align="center">

# Collaborative Reconstruction and Repair for Multi-class Industrial Anomaly Detection

### DI(Data Intelligence) 2025 
[![arXiv](https://img.shields.io/badge/arXiv-2405.14325-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2405.14325) [![CVF](https://img.shields.io/badge/CVPR-Paper-b4c7e7.svg?style=plastic)]

</div>

PyTorch Implementation of DI 2025 "Collaborative Reconstruction and Repair for Multi-class Industrial Anomaly Detection". 

**Give me a ⭐️ if you like it.**

<img width="2824" height="1575" alt="Figure2_1" src="https://github.com/user-attachments/assets/0ed5f847-43f6-439e-b439-17d46a066def" />


## News
 
 - _08.12.2025_: Accepted by CVPR 2025🎉

 - _15.12.2025_: Arxiv preprint and github code released🚀

## Abstract

Industrial anomaly detection is a challenging open-set task that aims to identify unknown anomalous patterns deviating from normal data distribution. To avoid the significant memory consumption and limited generalizability brought by building separate models per class, we focus on developing a unified framework for multi-class anomaly detection. However, under this challenging setting, conventional reconstruction-based networks often suffer from an identity mapping problem, where they directly replicate input features regardless of whether they are normal or anomalous, resulting in detection failures. To address this issue, this study proposes a novel framework termed Collaborative Reconstruction and Repair (CRR), which transforms the reconstruction to repairation. First, we optimize the decoder to reconstruct normal samples while repairing synthesized anomalies. Consequently, it generates distinct representations for anomalous regions and similar representations for normal areas compared to the encoder's output. Second, we implement feature-level random masking to ensure that the representations from decoder contain sufficient local information. Finally, to minimize detection errors arising from the discrepancies between feature representations from the encoder and decoder, we train a segmentation network supervised by synthetic anomaly masks, thereby enhancing localization performance. Extensive experiments on industrial datasets that CRR effectively mitigates the identity mapping issue and achieves state-of-the-art performance in multi-class industrial anomaly detection.

## 1. Environments

Create a new conda environment and install required packages.

```
conda create -n my_env python=3.8.12
conda activate my_env
pip install -r requirements.txt
```
Experiments are conducted on NVIDIA GeForce RTX 3090 (24GB). Same GPU and package version are recommended. 

## 2. Prepare Datasets
Noted that `../` is the upper directory of Dinomaly code. It is where we keep all the datasets by default.
You can also alter it according to your need, just remember to modify the `data_path` in the code. 

### MVTec AD

Download the MVTec-AD dataset from [URL](https://www.mvtec.com/company/research/datasets/mvtec-ad).
Unzip the file to `../mvtec_anomaly_detection`.
```
|-- mvtec_anomaly_detection
    |-- bottle
    |-- cable
    |-- capsule
    |-- ....
```


### VisA

Download the VisA dataset from [URL](https://github.com/amazon-science/spot-diff).
Unzip the file to `../VisA/`. Preprocess the dataset to `../VisA_pytorch/` in 1-class mode by their official splitting 
[code](https://github.com/amazon-science/spot-diff).

You can also run the following command for preprocess, which is the same to their official code.

```
python ./prepare_data/prepare_visa.py --split-type 1cls --data-folder ../VisA --save-folder ../VisA_pytorch --split-file ./prepare_data/split_csv/1cls.csv
```
`../VisA_pytorch` will be like:
```
|-- VisA_pytorch
    |-- 1cls
        |-- candle
            |-- ground_truth
            |-- test
                    |-- good
                    |-- bad
            |-- train
                    |-- good
        |-- capsules
        |-- ....
```
 
### Real-IAD
Contact the authors of Real-IAD [URL](https://realiad4ad.github.io/Real-IAD/) to get the net disk link.

Download and unzip `realiad_1024` and `realiad_jsons` in `../Real-IAD`.
`../Real-IAD` will be like:
```
|-- Real-IAD
    |-- realiad_1024
        |-- audiokack
        |-- bottle_cap
        |-- ....
    |-- realiad_jsons
        |-- realiad_jsons
        |-- realiad_jsons_sv
        |-- realiad_jsons_fuiad_0.0
        |-- ....
```

## 3. Run Experiments
Training Stage I
```
python dinomaly_realiad_uni_first.py --data_path ../Real-IAD
```
Training Phase II
```
python dinomaly_realiad_uni_second.py --data_path ../Real-IAD
```
Testing phase
```
python dinomaly_realiad_uni_test.py --data_path ../Real-IAD
```

### Trained model weights
| Dataset                | Model | Resolution  | Iterations  | Download        |
|----------------------|------------|------------|-------|--------------------|
| Real-IAD | Seg_model_7999.pth | R448-C392 |8,000 | [Google Drive](https://drive.google.com/file/d/1RXTmMDsE7TjRtTMDIXO0812xDp2vJ_MW/view?usp=sharing)  |
| Real-IAD | dinov2_vitb14_reg4_pretrain.pth | R448-C392 | 50,000 | [Google Drive](https://drive.google.com/file/d/1dZxqiKxApo6fDLiSA-vTLr7aOfMIr3JL/view?usp=sharing) |


## Results

**A. Compare with MUAD SOTAs:**
<div align="center">

<img alt="image" src="https://github.com/user-attachments/assets/082922bb-e8f8-4efc-9597-2a7dc8577d6e" />

<img width="869" height="482" alt="image" src="https://github.com/user-attachments/assets/9da30ae7-5c7f-4117-ad93-bf12f0fd98f0" />

</div>



**Dinomaly can perfectly scale with model size, input image size, and the choice of foundation model.**

**B. Model Size:**
<div align="center">

<img width="400" height="220" alt="image" src="https://github.com/user-attachments/assets/6f388ab7-0b81-450b-ae13-358a00c74f3f" />

<img width="865" height="190" alt="image" src="https://github.com/user-attachments/assets/a5d7c83f-bc64-4704-8607-a7a00cffe545" />
<img width="700" alt="image" src="https://github.com/user-attachments/assets/5005caed-2294-4766-92ed-ee93df5c5428" />

</div>


**C. Input Size:**
<div align="center">

<img width="400" height="220" alt="image" src="https://github.com/user-attachments/assets/e9a324a3-7f26-4d69-8806-a183042a3388" />

<img width="865" height="302" alt="image" src="https://github.com/user-attachments/assets/4f259320-2e4b-4796-aa7e-740bbd246d37" />

</div>

**D. Choice of Foundaiton Model:**
<div align="center">

<img width="400" height="220" alt="image" src="https://github.com/user-attachments/assets/a1ae0beb-ac5d-4926-94d4-4a99e07de03b" />

<img width="865" height="474" alt="image" src="https://github.com/user-attachments/assets/8c95f29b-578e-481d-bf0c-75429f76158f" />

</div>


## Eval discrepancy of anomaly localization
In our code implementation, we binarize the GT mask using gt.bool() after down-sampling, specifically gt[gt>0]=1. As pointed out in an issue, the previous common practice is to use gt[gt>0.5]=1. 
The difference between these two binarization approaches is that gt[gt>0]=1 may result in anomaly regions being one pixel larger compared to gt[gt>0.5]=1. This difference does not affect image-level performance metrics, but it has a slight impact on pixel-level evaluation metrics. 

We think gt[gt>0]=1 is a more reasonable choice. It can be seen as max pooling, so that in the down-sampled GT map, any position that corresponds to a region containing at least one anomaly pixel in the original map is marked as anomalous. If an anomaly region is extremely small in the original image (say 2 pixels), gt[gt>0.5]=1 will erase it while gt[gt>0]=1 can keep it.

## Loss NaN
If you encounter Loss=NaN during training on other datasets (very rare in common datasets), simply add a small eps (1e-6 by default) in the LinearAttention2 module:

`        z = 1.0 / (torch.einsum('...sd,...d->...s', q, k.sum(dim=-2)) + self.eps)
`

## Citation
```
@inproceedings{guo2025dinomaly,
  title={Dinomaly: The less is more philosophy in multi-class unsupervised anomaly detection},
  author={Guo, Jia and Lu, Shuai and Zhang, Weihang and Chen, Fang and Li, Huiqi and Liao, Hongen},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={20405--20415},
  year={2025}
}

@article{guo2025one,
  title={One Dinomaly2 Detect Them All: A Unified Framework for Full-Spectrum Unsupervised Anomaly Detection},
  author={Guo, Jia and Lu, Shuai and Fan, Lei and Li, Zelin and Di, Donglin and Song, Yang and Zhang, Weihang and Zhu, Wenbing and Yan, Hong and Chen, Fang and Li, Huiqi and Liao, Hongen},
  journal={arXiv preprint arXiv:2510.17611},
  year={2025}
}

```

