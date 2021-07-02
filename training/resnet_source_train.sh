sourceDataset='RAF'
targetDataset='AISIN'
pretrained='True'
class_num=2
n_source_train=2465
n_target_train=1700

Backbone='ResNet50'
epochs=60
useLocalFeature='True'
lr=0.0001
saveCheckPoint='False'

OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=${1} python3 TrainOnSourceDomain.py \
--log $2 \
--out ./exp_logs/$2 \
--pretrained ${pretrained} \
--net ${Backbone} \
--source ${sourceDataset} \
--target ${targetDataset} \
--epochs ${epochs} \
--lr ${lr} \
--class_num ${class_num} \
--local_feat ${useLocalFeature} \
--save_checkpoint ${saveCheckPoint} \
--source_labeled ${n_source_train} \
--target_unlabeled ${n_target_train}
