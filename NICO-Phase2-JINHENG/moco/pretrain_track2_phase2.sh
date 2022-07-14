arch=$1
data_dir=$2
track2='/Track2/public_ood_0412_nodomainlabel/train/'

echo "backbone: $arch"
python3 main_moco_pretraining_phase2.py \
  -a $arch \
  --workers 32 \
  --lr 0.03 \
  --batch-size 128 --epochs 200 \
  --input-size 224 \
  --dist-url 'tcp://localhost:10005' --multiprocessing-distributed --world-size 1 --rank 0 \
  --gpus 0,1,2,3 \
  --mlp --moco-t 0.2 --moco-k 4096 --aug-plus --cos \
  --track track2 \
  $data_dir$track2