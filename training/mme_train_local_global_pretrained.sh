OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=$1 python Train_MME.py  --log $2  --out ./exp_logs/$2  --local_feat True --pretrained ./pretrained_models/ckpts/ir50_local_2class.pkl
