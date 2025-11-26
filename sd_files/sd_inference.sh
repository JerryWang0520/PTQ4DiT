# CIFAR-10 (DDIM)
# 4/8-bit weights-only
python scripts/sample_diffusion_ddim.py \
    --config configs/cifar10.yml \
    --use_pretrained \
    --timesteps 100 \
    --eta 0 \
    --skip_type quad \
    --ptq \
    --weight_bit <4 or 8> \
    --quant_mode qdiff \
    --split \
    --resume \
    -l <output_path> \
    --cali_ckpt <quantized_ckpt_path>

# 4/8-bit weights, 8-bit activations
python scripts/sample_diffusion_ddim.py \
    --config configs/cifar10.yml \
    --use_pretrained \
    --timesteps 100 \
    --eta 0 \
    --skip_type quad \
    --ptq \
    --weight_bit <4 or 8> \
    --quant_mode qdiff \
    --quant_act \
    --act_bit 8 \
    --a_sym \
    --split \
    --resume \
    -l <output_path> \
    --cali_ckpt <quantized_ckpt_path>

# LSUN Bedroom (LDM-4)
# 4/8-bit weights-only
python scripts/sample_diffusion_ldm.py \
    -r models/ldm/lsun_beds256/model.ckpt \
    -n 20 \
    --batch_size 10 \
    -c 200 \
    -e 1.0 \
    --seed 41 \
    --ptq \
    --weight_bit <4 or 8> \
    --resume \
    -l <output_path> \
    --cali_ckpt <quantized_ckpt_path>

# 4/8-bit weights, 8-bit activations
python scripts/sample_diffusion_ldm.py \
    -r models/ldm/lsun_beds256/model.ckpt \
    -n 20 \
    --batch_size 10 \
    -c 200 \
    -e 1.0 \
    --seed 41 \
    --ptq \
    --weight_bit <4 or 8> \
    --quant_act \
    --act_bit 8 \
    --a_sym \
    --resume \
    -l <output_path> \
    --cali_ckpt <quantized_ckpt_path>

# LSUN Church (LDM-8)
# 4/8-bit weights-only
python scripts/sample_diffusion_ldm.py \
    -r models/ldm/lsun_churches256/model.ckpt \
    -n 20 \
    --batch_size 10 \
    -c 400 \
    -e 0.0 \
    --seed 41 \
    --ptq \
    --weight_bit <4 or 8> \
    --resume \
    -l <output_path> \
    --cali_ckpt <quantized_ckpt_path>

# 4/8-bit weights, 8-bit activations
python scripts/sample_diffusion_ldm.py \
    -r models/ldm/lsun_churches256/model.ckpt \
    -n 20 \
    --batch_size 10 \
    -c 400 \
    -e 0.0 \
    --seed 41 \
    --ptq \
    --weight_bit <4 or 8> \
    --quant_act \
    --act_bit 8 \
    --resume \
    -l <output_path> \
    --cali_ckpt <W8A8 quantized_ckpt_path>

# Stable Diffusion
# 4/8-bit weights-only
python scripts/txt2img.py \
    --prompt <prompt. e.g. "a puppy wearing a hat"> \
    --plms \
    --cond \
    --ptq \
    --weight_bit <4 or 8> \
    --quant_mode qdiff \
    --no_grad_ckpt \
    --split \
    --n_samples 5 \
    --resume \
    --outdir <output_path> \
    --cali_ckpt <quantized_ckpt_path>

# 4/8-bit weights, 8-bit activations (with 16-bit for attention matrices after softmax)
python scripts/txt2img.py \
    --prompt <prompt. e.g. "a puppy wearing a hat"> \
    --plms \
    --cond \
    --ptq \
    --weight_bit <4 or 8> \
    --quant_mode qdiff \
    --no_grad_ckpt \
    --split \
    --n_samples 5 \
    --resume \
    --quant_act \
    --act_bit 8 \
    --sm_abit 16 \
    --outdir <output_path> \
    --cali_ckpt <quantized_ckpt_path>