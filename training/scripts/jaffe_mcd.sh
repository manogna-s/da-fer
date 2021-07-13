sourceDataset='RAF_7class'
targetDataset='JAFFE'
class_num=7
n_source_train=-1
n_target_train=-1
useLocalFeature='True'
saveCheckPoint='False'

OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=$1 nohup python Train_MCD.py  \
--use_mcd True \
--use_grl False \
--log summary_logs  \
--out ./exp_logs/$2 \
--local_feat ${useLocalFeature} \
--pretrained True \
--source ${sourceDataset} \
--target ${targetDataset} \
--class_num ${class_num} \
--save_checkpoint ${saveCheckPoint} \
--source_labeled ${n_source_train} \
--target_unlabeled ${n_target_train} \
--train_batch 32 >./exp_logs/$2_training_log.txt
