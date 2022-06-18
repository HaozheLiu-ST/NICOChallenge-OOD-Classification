CUDA_VISIBLE_DEVICES=1 python model_soup.py \
-a densenet121 \
--ckpts \
/apdcephfs/share_1290796/waltszhang/NICO_challenge/train_dg_ddp/ckpts/epoch_200_60epoch_FDG_512_1e-1_coslr_ratio_0.2/NICO-densenet121-track2/teacher/model_best_out_090_epochs.pth.tar \
/apdcephfs/share_1290796/haozheliu/NICO/Model_Soup/ckpts/ratio_05_lr_1e5NICO-densenet121-track2/student/model_best_in_090_epochs.pth.tar \
/apdcephfs/share_1290796/haozheliu/NICO/Model_Soup/ckpts/ratio_02_lr_1e4NICO-densenet121-track2/student/model_best_in_090_epochs.pth.tar \
/apdcephfs/share_1290796/haozheliu/NICO/Model_Soup/ckpts/ratio_05_lr_1e3NICO-densenet121-track2/student/model_best_in_090_epochs.pth.tar \
/apdcephfs/share_1290796/haozheliu/NICO/Model_Soup/ckpts/ratio_06_lr_1e3NICO-densenet121-track2/student/model_best_in_090_epochs.pth.tar \
/apdcephfs/share_1290796/haozheliu/NICO/Model_Soup/ckpts/ratio_04_lr_1e3NICO-densenet121-track2/student/model_best_in_090_epochs.pth.tar \
/apdcephfs/share_1290796/haozheliu/NICO/Model_Soup/ckpts/ratio_05_lr_1e4NICO-densenet121-track2/student/model_best_in_090_epochs.pth.tar \
/apdcephfs/share_1290796/haozheliu/NICO/Model_Soup/ckpts/ratio_08_lr_1e3NICO-densenet121-track2/student/model_best_in_090_epochs.pth.tar \
/apdcephfs/share_1290796/haozheliu/NICO/Model_Soup/ckpts/ratio_01_lr_1e4NICO-densenet121-track2/student/model_best_in_090_epochs.pth.tar \
--root '/apdcephfs/share_1290796/Datasets/NICO/nico_datasets/' \
--track 2 \
--momentum 0.9 \
--save_path ./ckpts/model_soup_greedy_submodel/