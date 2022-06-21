python3 main_moco_pretraining_with_domain.py \
  -a densenet121 \
  --workers 32 \
  --lr 0.03 \
  --batch-size 128 --epochs 200 \
  --input-size 224 \
  --dist-url 'tcp://localhost:10004' --multiprocessing-distributed --world-size 1 --rank 0 \
  --gpus 1,2,3,4 \
  --mlp --moco-t 0.2 --moco-k 4096 --aug-plus --cos \
  --track track1 \
  /apdcephfs/share_1290796/jinhengxie/datasets/public_dg_0416/train_domain_free
