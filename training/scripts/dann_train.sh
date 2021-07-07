sourceDataset='RAF'
targetDataset='AISIN'
class_num=2
n_source_train=2465
n_target_train=1700
useLocalFeature='True'
saveCheckPoint='False'

OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=$1 python TransferToTargetDomain.py  \
--log $2 \
--out ./exp_logs/$2 \
--local_feat True \
--pretrained True \
--use_dan True \
--dan_method $3 \
--source ${sourceDataset} \
--target ${targetDataset} \
--class_num ${class_num} \
--local_feat ${useLocalFeature} \
--save_checkpoint ${saveCheckPoint} \
--source_labeled ${n_source_train} \
--target_unlabeled ${n_target_train}
~
