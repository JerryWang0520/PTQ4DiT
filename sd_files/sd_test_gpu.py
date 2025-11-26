import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# AssertionError: Torch not compiled with CUDA enabled
# conda remove pytorch torchvision torchaudio
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia