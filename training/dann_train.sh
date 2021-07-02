sourceDataset='RAF'
targetDataset='AISIN'
class_num=2
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=$1 python TransferToTargetDomain.py  \
--log $2 \
--out ./exp_logs/$2 \
--local_feat True \
--pretrained True \
--use_dan True \
--dan_method $3 \
--source ${sourceDataset} \
--target ${targetDataset} \
--class_num ${class_num}

