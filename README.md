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
    - Source: 'RAF', 'RAF_2class' (use 'RAF' for 7class, 'RAF_2class' for 2class)
    - Target: 'AISIN', 'JAFFE', 'CK+'
- To use other backbones/ datasets refer [AGRA official code](https://github.com/HCPLab-SYSU/CD-FER-Benchmark/tree/master/AGRA)    

## RAF_2class to AISIN 
Classes: Happy(Class 0), Neutral(Class 1)

### Dataset and pretrained models
Place data and ckpts in this folder structure
```commandline
+ da-fer
    + Dataset
        + RAF
            + Train
                + Anger
                + Disgust
                + Fear
                + Happy
                + Neutral
                + Sad
                + Surprised
            + Test
                + Anger
                .
                .
            + annotations
                + train_annotations.json
                + test_annotations.json
        + AISIN
            + Train
                + Happy
                + Neutral
            + Test
                + Happy
                + Neutral
            + annotations
                + train_annotations.json
                + test_annotations.json
            
    + pretrained_ckpts
        + backbone_ir50_ms1m_epoch120.pth
```


### Prepare data
Organize as above and run this to get annotation with bbox and landmarks.
```commandline
cd Dataset/AISIN
python get_crops.py
```

## Training

 Refer [training](training/README.md)





