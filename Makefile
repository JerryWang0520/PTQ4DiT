
run_original:
	python quant_sample.py \
		--model DiT-XL/2 \
		--image-size 256 \
		--ckpt pretrained_models/DiT-XL-2-256x256.pt \
		--num-sampling-steps 50 \
		--weight_bit 8 \
		--act_bit 8 \
		--cali_st 25 \
		--cali_n 64 \
		--cali_batch_size 32 \
		--sm_abit 8 \
		--cali_data_path calib_2/imagenet_DiT-256_sample4000_50steps_allst.pt \
		--outdir output/ \
		--cfg-scale 1.5 \
		--seed 1 \
		--resume \
		--cali_ckpt ./cali_ckpt_2/256_88_50/ckpt.pth \
		--ptq \
		--inference \
		--n_c 1 \
		--c_begin 205 \
		--c_end 205

run_spat:
	python quant_sample.py \
		--model DiT-XL/2 \
		--image-size 256 \
		--ckpt pretrained_models/DiT-XL-2-256x256.pt \
		--num-sampling-steps 50 \
		--weight_bit 8 \
		--act_bit 8 \
		--cali_st 25 \
		--cali_n 64 \
		--cali_batch_size 32 \
		--sm_abit 8 \
		--cali_data_path calib_2/imagenet_DiT-256_sample4000_50steps_allst.pt \
		--outdir output/ \
		--cfg-scale 1.5 \
		--seed 1 \
		--resume \
		--cali_ckpt ./cali_ckpt_2/256_88_50/ckpt.pth \
		--ptq \
		--inference \
		--n_c 1 \
		--c_begin 205 \
		--c_end 205 \
		--reuse_type=spatial


run_temp:
	python quant_sample.py \
		--model DiT-XL/2 \
		--image-size 256 \
		--ckpt pretrained_models/DiT-XL-2-256x256.pt \
		--num-sampling-steps 50 \
		--weight_bit 8 \
		--act_bit 8 \
		--cali_st 25 \
		--cali_n 64 \
		--cali_batch_size 32 \
		--sm_abit 8 \
		--cali_data_path calib_2/imagenet_DiT-256_sample4000_50steps_allst.pt \
		--outdir output/ \
		--cfg-scale 1.5 \
		--seed 1 \
		--resume \
		--cali_ckpt ./cali_ckpt_2/256_88_50/ckpt.pth \
		--ptq \
		--inference \
		--n_c 1 \
		--c_begin 205 \
		--c_end 205 \
		--reuse_type=temporal

run_cfg:
	python quant_sample.py \
		--model DiT-XL/2 \
		--image-size 256 \
		--ckpt pretrained_models/DiT-XL-2-256x256.pt \
		--num-sampling-steps 50 \
		--weight_bit 8 \
		--act_bit 8 \
		--cali_st 25 \
		--cali_n 64 \
		--cali_batch_size 32 \
		--sm_abit 8 \
		--cali_data_path calib_2/imagenet_DiT-256_sample4000_50steps_allst.pt \
		--outdir output/ \
		--cfg-scale 1.5 \
		--seed 1 \
		--resume \
		--cali_ckpt ./cali_ckpt_2/256_88_50/ckpt.pth \
		--ptq \
		--inference \
		--n_c 1 \
		--c_begin 205 \
		--c_end 205 \
		--reuse_type=cfg
