# CIFAR-10 (DDIM)
python scripts/sample_diffusion_ddim.py \
    --config configs/cifar10.yml \
    --use_pretrained \
    --timesteps 100 \
    --eta 0 \
    --skip_type quad \
    --ptq \
    --weight_bit <4 or 8> \
    --quant_mode qdiff \
    --cali_st 20 \
    --cali_batch_size 32 \
    --cali_n 256 \
    --quant_act \
    --act_bit 8 \
    --a_sym \
    --split \
    --cali_data_path <cali_data_path> \
    -l <output_path>

# LSUN Bedroom (LDM-4)
python scripts/sample_diffusion_ldm.py \
    -r models/ldm/lsun_beds256/model.ckpt \
    -n 50000 \
    --batch_size 10 \
    -c 200 \
    -e 1.0 \
    --seed 40 \
    --ptq \
    --weight_bit <4 or 8> \
    --quant_mode qdiff \
    --cali_st 20 \
    --cali_batch_size 32 \
    --cali_n 256 \
    --quant_act \
    --act_bit 8 \
    --a_sym \
    --a_min_max \
    --running_stat \
    --cali_data_path <sample cali_data_path> \
    -l <output_path>

# LSUN Church (LDM-8)
python scripts/sample_diffusion_ldm.py \
    -r models/ldm/lsun_churches256/model.ckpt \
    -n 50000 \
    --batch_size 10 \
    -c 400 \
    -e 0.0 \
    --seed 40 \
    --ptq \
    --weight_bit <4 or 8> \
    --quant_mode qdiff \
    --cali_st 20 \
    --cali_batch_size 32 \
    --cali_n 256 \
    --quant_act \
    --act_bit 8 \
    --cali_data_path <cali_data_path> \
    -l <output_path>

# Stable Diffusion
python scripts/txt2img.py \
    --prompt "a photograph of an astronaut riding a horse" \
    --plms \
    --cond \
    --ptq \
    --weight_bit <4 or 8> \
    --quant_mode qdiff \
    --quant_act \
    --act_bit 8 \
    --cali_st 25 \
    --cali_batch_size 8 \
    --cali_n 128 \
    --no_grad_ckpt \
    --split \
    --running_stat \
    --sm_abit 16 \
    --cali_data_path <cali_data_path> \
    --outdir <output_path>