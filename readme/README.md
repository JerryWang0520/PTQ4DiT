# Diffusion models with Differential Computing and Bit-Slice Computing

## Overview

This project extends the original [PTQ4DiT](https://github.com/adreamwu/PTQ4DiT) and [q-diffusion](https://github.com/Xiuyu-Li/q-diffusion) with differential computing and bit-slice computing.

## Added Features

- **Differential Computing**: Multiple methods for optimized quantized operations
    - Raw + Raw
    - SDC + SDC
    - TDC + TDC
    - SDC + GDC
    - Raw + SDC
    - Raw + TDC
    - Raw + GDC
    - SDC + TDC
    - TDC + GDC
    - GDC_large
    - GDC_opt
    - SDC + GDC_opt

- **Bit-Slice Computing**: Multiple strategies for efficient low-bit arithmetic
    - Naive bit-slice
    - Sibia
    - BADA

## Usage

All commands are available in `script.sh` for reference.

**Note**: For q-diffusion evaluation, copy files from `sd_files/` to the q-diffusion project root.


### Similarity
Compare activation patterns between different activation differences:
```bash
make run_strategy analysis=similarity

# Example:
# make run_strategy hook=True strategy=original analysis=similarity similarity_type=spatial
```

### Bitwidth
Analyze bitwidth requirements across layers:
```bash
make run_strategy analysis=bitwidth

# Example:
# make run_strategy outdir=output/bitwidth/Raw n_c=1 c_begin=0 c_end=0 hook=True strategy=original analysis=bitwidth path_reverse=False
```

### Bit-Slice Amount
Evaluate the number of bit-slices needed:
```bash
make run_strategy bitslice_method=<BIT-SLICE METHOD> performance=False
make summarize_bitslice indir=<INPUT DIRECTORY> outdir=<OUTPUT DIRECTORY>

# Example:
# make run_strategy outdir=output/bitslice/bada_int9_bs3/Raw  n_c=1 c_begin=0 c_end=0 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=False performance=False
# make summarize_bitslice indir=output/bitslice/bada_int9_bs3/Raw/Final  outdir=output/summary/bitslice/bada_int9_bs3/Raw  bitslice_overflow=True
```

### Performance
Measure computational performance:
```bash
make performance bitslice_method=<BIT-SLICE METHOD> performance=True
make summarize_performance indir=<INPUT DIRECTORY> outdir=<OUTPUT DIRECTORY>

# Example:
# make run_strategy outdir=output/performance/bada_int9_bs3/original/scheme1  n_c=1 c_begin=0 c_end=0 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=False performance=True
# make summarize_performance indir=output/performance/bada_int9_bs3/original/scheme1/Final outdir=output/summary/performance/bada_int9_bs3/original/scheme1
```

### FID Evaluation
Compute FID scores for image quality assessment:
```bash
make run_samples
python evaluations/compress.py --sample-dir=<SAMPLE DIRECTORY> --num-fid-samples=<NUM SAMPLES>
source set_env.sh
python evaluations/evaluator.py <COMPRESSED REFERENCE FILE> <COMPRESSED SAMPLE FILE>

# Example:
# make run_samples outdir=output/bitslice/bada_int9_bs3/original n_c=10 c_begin=0   c_end=199 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# python evaluations/compress.py --sample-dir=output/samples/inference --num-fid-samples=10000
# python evaluations/evaluator.py eval_batch/VIRTUAL_imagenet256_labeled.npz output/samples/inference.npz
```

Download reference batches from the [guided-diffusion evaluations](https://github.com/openai/guided-diffusion/tree/main/evaluations).

### TensorFlow CuDNN Version Mismatch Error

When running the FID evaluation script, you may encounter a CuDNN version mismatch error. The error occurs because TensorFlow finds the system's CuDNN library (version 9.1.0) instead of the version it was compiled with (version 9.3.0), resulting in "No DNN in stream executor" errors.

The root cause is that the dynamic linker searches for shared libraries in system paths before checking the conda environment. Although image generation with PyTorch works correctly, TensorFlow's FID evaluation fails because it cannot properly load the matching CuDNN version.

To resolve this issue, you need to set the `LD_LIBRARY_PATH` environment variable to prioritize the conda environment's library directory. This ensures TensorFlow loads the correct CuDNN version from your conda environment rather than the system installation.

**Solution**: Before running the FID evaluation, source the environment setup script:

```bash
source set_env.sh
```

This script sets `LD_LIBRARY_PATH` to include your conda environment's lib directory at the highest priority, allowing TensorFlow to find and use the compatible CuDNN library. You need to source this script each time you start a new terminal session before running evaluations.

### Hardware Pattern Generation
Hook and save computation patterns:
```bash
make run_strategy save_tile=True
tar zcvf <FileName.tar.gz> <DirName>

# Example:
# make run_strategy outdir=output/patterns/original n_c=1 c_begin=0 c_end=0 hook=True strategy=original    save_tile=True path_reverse=False
```

Create test patterns:
```bash
python saver/tensor_saver.py
tar zcvf <FileName.tar.gz> <DirName>
```

## License

For internal research and development use.