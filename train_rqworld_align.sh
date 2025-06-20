CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --py-config config/train_cascadeworld_ms_3_rqvaealign.py --work-dir out/rqvae_increment_celoss
# CUDA_VISIBLE_DEVICES=0,1 python train.py --py-config config/train_cascadeworld_ms_3.py --work-dir out/newrqvae_stage2_indivcausal_nocumsum
# CUDA_VISIBLE_DEVICES=2,3 python train.py --py-config config/train_cascadeworld_ms_3.py --work-dir out/newrqvae_stage2_indivcausal_cumsum
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --py-config config/train_cascadeworld_ms_3.py --work-dir out/cascade_stage2_ms3_refined_aligned
