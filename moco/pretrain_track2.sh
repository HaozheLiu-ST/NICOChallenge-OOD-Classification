python3 main_moco_pretraining.py \
  -a mobilenet_v2 \
  --workers 32 \
  --lr 0.03 \
  --batch-size 128 --epochs 200 \
  --input-size 224 \
  --dist-url 'tcp://localhost:10004' --multiprocessing-distributed --world-size 1 --rank 0 \
  --gpus 1,2,3,4 \
  --mlp --moco-t 0.2 --moco-k 4096 --aug-plus --cos \
  --track track2 \
  /apdcephfs/share_1290796/Datasets/NICO/nico_datasets/Track2/public_ood_0412_nodomainlabel/train/