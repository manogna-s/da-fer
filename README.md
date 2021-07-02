#  Domain Adaptation for Facial Expression Recognition

This repository is built using some part of code from [here](https://github.com/HCPLab-SYSU/CD-FER-Benchmark). The original paper is linked [here](https://arxiv.org/abs/2008.00859)

7 expressions used in FER task in general:

```commandline
    0: Surprised
    1: Fear
    2: Disgust
    3: Happy
    4: Sad
    5: Angry
    6: Neutral
```

- This code supports only ResNet-18/50 backbones and is validated using only ResNet50.
- Datasets supported:
    - Source: 'RAF', 'RAF_7class' (use 'RAF' for 2class)
    - Target: 'AISIN', 'JAFFE'
- To use other backbones/ datasets refer [AGRA official code](https://github.com/HCPLab-SYSU/CD-FER-Benchmark/tree/master/AGRA)    

## RAF to AISIN 
Classes: Happy, Neutral

### Dataset and pretrained models
Place data and ckpts in this folder structure
```commandline
+ da-fer
    + Dataset
        + RAF
            + images
            + boundingbox
            + landmarks_5
            + lists
        + AISIN
            + images
            + boundingbox
            + landmarks_5
            + lists
            
    + pretrained_ckpts
        + backbone_ir50_ms1m_epoch120.pth
```


## Training

### Features
- Global(Face) + local(landmarks): Set ```--local_feat True```
- Only Global(Face)              : Set ```--local_feat False```

### Methods
- Train only on source data 
  - [resnet_source_train.sh](training/resnet_source_train.sh)  
  - [gcn_source_train.sh](training/gcn_source_train.sh)
- GCN with DANN variants: DANN, CDAN, CDAN-E
  - [gcn_target_train.sh](training/gcn_target_train.sh)
- MME
  - [mme_train.sh](training/mme_train.sh)
    
Check ```training/train_setup.py``` for entire list of arguments. Edit the *.sh files to configure arguments as required.
To run, ```sh file_to_run.sh GPU_ID exp_name```. Eg.,

```
cd training
sh gcn_source_train.sh 0 exp_gcn_source
```

To set number of samples per class/ to use balanced dataset (only for 2-class RAF to AISIN) , use args 
```
--source_labeled {n_source_train} 
--target_unlabeled {n_target_train}
```
For RAF to AISIN with balanced dataset, use approx n_source_train = 2465, n_target_train = 1700
 
Note: Training GCN on source and using k-means init is not working currently. 





