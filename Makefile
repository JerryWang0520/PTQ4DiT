
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
		$(if $(outdir),--outdir=$(outdir)) \
		--cfg-scale 1.5 \
		--seed 1 \
		--resume \
		--cali_ckpt ./cali_ckpt/256_88_50/ckpt.pth \
		--ptq \
		--inference \
		--n_c $(n_c) \
		--c_begin $(c_begin) \
		--c_end $(c_end) \
		$(if $(filter True,$(sample)),--sample) \
		$(if $(filter True,$(hook)),--hook) \
		$(if $(strategy),--strategy=$(strategy)) \
		$(if $(filter True,$(ref_first)),--ref_first) \
		$(if $(bit_th),--bit_th=$(bit_th)) \
		$(if $(filter True,$(clamp)),--clamp) \
		$(if $(analysis),--analysis=$(analysis)) \
		$(if $(filter similarity,$(analysis)),--similarity-types=$(similarity_type)) \
		$(if $(bitslice_method),--bitslice-method=$(bitslice_method)) \
		$(if $(bitslice_bits),--bitslice-bits=$(bitslice_bits)) \
		$(if $(bitslice_width),--bitslice-width=$(bitslice_width)) \
		$(if $(filter True,$(bitslice_overflow)),--bitslice-keep-overflow) \
		$(if $(filter True,$(bitslice_clamp)),--bitslice-clamp) \
		$(if $(scheme),--scheme=$(scheme)) \
		$(if $(filter True,$(path_reverse)),--path-reverse) \
		$(if $(filter True,$(save_tile)),--save-tile) \
		$(if $(filter True,$(performance)),--performance) \

run_samples:
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
		$(if $(outdir),--outdir=$(outdir)) \
		--cfg-scale 1.5 \
		--seed 1 \
		--resume \
		--cali_ckpt ./cali_ckpt/256_88_50/ckpt.pth \
		--ptq \
		--inference \
		--n_c $(n_c) \
		--c_begin $(c_begin) \
		--c_end $(c_end) \
		--sample \
		$(if $(filter True,$(hook)),--hook) \
		$(if $(strategy),--strategy=$(strategy)) \
		$(if $(filter True,$(ref_first)),--ref_first) \
		$(if $(bit_th),--bit_th=$(bit_th)) \
		$(if $(filter True,$(clamp)),--clamp) \
		$(if $(bitslice_method),--bitslice-method=$(bitslice_method)) \
		$(if $(bitslice_bits),--bitslice-bits=$(bitslice_bits)) \
		$(if $(bitslice_width),--bitslice-width=$(bitslice_width)) \
		$(if $(filter True,$(bitslice_overflow)),--bitslice-keep-overflow) \
		$(if $(filter True,$(bitslice_clamp)),--bitslice-clamp) \
		$(if $(scheme),--scheme=$(scheme)) \
		$(if $(filter True,$(path_reverse)),--path-reverse) \
		$(if $(filter True,$(save_tile)),--save-tile) \
		$(if $(filter True,$(performance)),--performance) \

summarize_bitwidth:
	mkdir -p $(outdir)
	python summaries/bitwidth_summary.py \
		--input-dir $(indir) \
		--output-dir $(outdir) \
		--distribution-pattern "class*/bitwidth/int9_full_distribution_act.csv" \
		--bitwidth-pattern "class*/bitwidth/bitwidth_act.csv" \
		--save \
		--stats \
		| tee $(outdir)/bitwidth_summary.log

summarize_bitslice:
	mkdir -p $(outdir)
	python summaries/bitslice_summary.py \
		--input-dir $(indir) \
		--output-dir $(outdir) \
		--pattern "class*/bitslice/bitslice_analysis.csv" \
		--save \
		--stats \
		$(if $(filter True,$(bitslice_overflow)),--bitslice-keep-overflow) \
		| tee $(outdir)/bitslice_summary.log

summarize_performance:
	mkdir -p $(outdir)
	python summaries/performance_summary.py \
		--input-dir $(indir) \
		--pattern "class*/performance/performance_analysis.csv" \
		--output-dir $(outdir) \
		--save \
		--stats \
		| tee $(outdir)/performance_summary.log

summarize_memory:
	python summaries/memory_summary.py | tee output/summary/memory_summary.log

test_bitslice:
	python computations/bitslice_strategies.py | tee output/test_bitslice.log
