
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
		--cali_data_path calib/imagenet_DiT-256_sample4000_50steps_allst.pt \
		--outdir output/ \
		--cfg-scale 1.5 \
		--seed 1 \
		--resume \
		--cali_ckpt ./cali_ckpt/256_88_50/ckpt.pth \
		--ptq \
		--inference \
		--n_c 1 \
		--c_begin 0 \
		--c_end 0 \
		--original

run_SD:
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
		--cali_data_path calib/imagenet_DiT-256_sample4000_50steps_allst.pt \
		--outdir output/ \
		--cfg-scale 1.5 \
		--seed 1 \
		--resume \
		--cali_ckpt ./cali_ckpt/256_88_50/ckpt.pth \
		--ptq \
		--inference \
		--n_c 1 \
		--c_begin 0 \
		--c_end 0 \
		--SD

run_TD:
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
		--cali_data_path calib/imagenet_DiT-256_sample4000_50steps_allst.pt \
		--outdir output/ \
		--cfg-scale 1.5 \
		--seed 1 \
		--resume \
		--cali_ckpt ./cali_ckpt/256_88_50/ckpt.pth \
		--ptq \
		--inference \
		--n_c 1 \
		--c_begin 0 \
		--c_end 0 \
		--TD

run_CUD:
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
		--cali_data_path calib/imagenet_DiT-256_sample4000_50steps_allst.pt \
		--outdir output/ \
		--cfg-scale 1.5 \
		--seed 1 \
		--resume \
		--cali_ckpt ./cali_ckpt/256_88_50/ckpt.pth \
		--ptq \
		--inference \
		--n_c 1 \
		--c_begin 0 \
		--c_end 0 \
		--CUD

run_quant_info:
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
		--cali_data_path calib/imagenet_DiT-256_sample4000_50steps_allst.pt \
		--outdir output/ \
		--cfg-scale 1.5 \
		--seed 1 \
		--resume \
		--cali_ckpt ./cali_ckpt/256_88_50/ckpt.pth \
		--ptq \
		--inference \
		--n_c 1 \
		--c_begin 0 \
		--c_end 0 \
        --quant_info

run_strategy:
	python quant_sample_v2.py \
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
		--cali_data_path calib/imagenet_DiT-256_sample4000_50steps_allst.pt \
		--outdir output/$(analysis) \
		--cfg-scale 1.5 \
		--seed 1 \
		--resume \
		--cali_ckpt ./cali_ckpt/256_88_50/ckpt.pth \
		--ptq \
		--inference \
		--n_c 1 \
		--c_begin 0 \
		--c_end 0 \
		$(if $(strategy),--strategy=$(strategy)) \
		$(if $(analysis),--analysis=$(analysis)) \
		$(if $(filter similarity, $(analysis)),--similarity-types=$(similarity_type))

# make run_strategy strategy=original analysis=bitwidth
# make run_strategy strategy=original analysis=similarity similarity_type=spatial
# make run_strategy strategy=original analysis=shape