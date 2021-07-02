sourceDataset='RAF'
targetDataset='AISIN'
class_num=2
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=$1 python Train_MME.py  \
--use_mme True \
--log $2  \
--out ./exp_logs/$2 \
--local_feat True \
--pretrained True \
--source ${sourceDataset} \
--target ${targetDataset} \
--class_num ${class_num}
