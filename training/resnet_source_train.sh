sourceDataset='RAF'
targetDataset='AISIN'
pretrained='True'
class_num=2
Backbone='ResNet50'
epochs=60
useLocalFeature='True'
lr=0.0001

OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=${1} python3 TrainOnSourceDomain.py \
--log $2 \
--out ./exp_logs/$2 \
--pretrained ${pretrained} \
--GPU_ID ${GPU_ID} \
--net ${Backbone} \
--source ${sourceDataset} \
--target ${targetDataset} \
--epochs ${epochs} \
--lr ${lr} \
--class_num ${class_num} \
--local_feat ${useLocalFeature} \
