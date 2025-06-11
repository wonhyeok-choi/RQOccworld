TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_VISIBLE_DEVICES=0,1,2,3 python train_vqvae.py --py-config config/train_rqvae.py --work-dir out/debug_rqvaeV2_stage1_depth_4_codebook_2048
# TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_VISIBLE_DEVICES=4,5 python train_vqvae.py --py-config config/train_vqvae.py --work-dir out/var_12_25_50_stage1_valid --resume-from out/var_12_25_50_stage1_valid/epoch_183.pth
