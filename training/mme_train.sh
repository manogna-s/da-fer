sourceDataset='RAF'
targetDataset='AISIN'
class_num=2
n_source_train=2465
n_target_train=1700
useLocalFeature='True'
saveCheckPoint='False'

OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=$1 python Train_MME.py  \
--use_mme True \
--log $2  \
--out ./exp_logs/$2 \
--local_feat ${useLocalFeature} \
--pretrained True \
--source ${sourceDataset} \
--target ${targetDataset} \
--class_num ${class_num} \
--save_checkpoint ${saveCheckPoint} \
--source_labeled ${n_source_train} \
--target_unlabeled ${n_target_train}
