<div align="center">

# Collaborative Reconstruction and Repair for Multi-class Industrial Anomaly Detection

### DI(Data Intelligence) 2025 
[![arXiv](https://img.shields.io/badge/arXiv-2405.14325-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2405.14325) 

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

<img width="1081" height="852" alt="table1" src="https://github.com/user-attachments/assets/d3030ee2-cfac-49fe-b824-d3b2af55ea64" />

</div>


## Citation
```

@article{guo2025one,
  title={One Dinomaly2 Detect Them All: A Unified Framework for Full-Spectrum Unsupervised Anomaly Detection},
  author={Guo, Jia and Lu, Shuai and Fan, Lei and Li, Zelin and Di, Donglin and Song, Yang and Zhang, Weihang and Zhu, Wenbing and Yan, Hong and Chen, Fang and Li, Huiqi and Liao, Hongen},
  journal={arXiv preprint arXiv:2510.17611},
  year={2025}
}

```

