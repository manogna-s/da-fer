## Training

### Features
- Global(Face) + local(landmarks): Set ```--local_feat True```
- Only Global(Face)              : Set ```--local_feat False```

### Methods
- Train only on source data 
  - [resnet_source_train.sh](scripts/resnet_source_train.sh)  
  - [gcn_source_train.sh](scripts/gcn_source_train.sh)
- GCN with DANN variants: DANN, CDAN, CDAN-E
  - [gcn_target_train.sh](scripts/gcn_target_train.sh)
- MME
  - [mme_train.sh](scripts/mme_train.sh)
- MCD

Check ```training/train_setup.py``` for entire list of arguments. Edit the *.sh files to configure arguments as required.
To run, ```sh file_to_run.sh GPU_ID exp_name```. Eg.,

```
cd training
sh scripts/gcn_source_train.sh 0 exp_gcn_source
```

To set number of samples per class/ to use balanced dataset (only for RAF_2class to AISIN) , use args 
```
--source_labeled {n_source_train} 
--target_unlabeled {n_target_train}
```
For RAF_2class to AISIN with balanced dataset, use approx n_source_train = 2465, n_target_train = 1700
 
Note: Training GCN on source and using k-means init is not working currently.