OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=$1 python Train_MME.py  --log $2  --out ./exp_logs/$2  --local_feat False --pretrained ./preTrainedModel/ir50_ms1m_112_onlyGlobal.pkl
