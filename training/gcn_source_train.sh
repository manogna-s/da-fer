pretrained='True'
Backbone='ResNet50'
sourceDataset='RAF'
targetDataset='AISIN'
n_source_train=2465
n_target_train=1700
useLocalFeature='True'
saveCheckPoint='False'
epochs=60
class_num=2
useClassify='True'
useLocalFeature='True'
useGCN='True'
useIntraGCN='True'
useInterGCN='True'
useCov='False'
useCluster='False'
lr=0.0001

OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=${1} python3 TrainOnSourceDomain.py \
--GPU_ID $1 \
--log $2 \
--out ./exp_logs/$2 \
--pretrained ${pretrained} \
--net ${Backbone} \
--source ${sourceDataset} \
--target ${targetDataset} \
--save_checkpoint ${saveCheckPoint} \
--source_labeled ${n_source_train} \
--target_unlabeled ${n_target_train} \
--epochs ${epochs} \
--lr ${lr} \
--class_num ${class_num} \
--useClassify ${useClassify} \
--use_gcn ${useGCN} \
--intra_gcn ${useIntraGCN} \
--inter_gcn ${useInterGCN} \
--local_feat ${useLocalFeature} \
--use_cov ${useCov} \
--use_cluster ${useCluster}
