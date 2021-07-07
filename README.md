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

 Refer [training](training/README.md)





