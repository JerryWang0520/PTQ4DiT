import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import os

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:0d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    # if folder_name is intended to be a variable, you need to define it.
    # For now, I assume sample_dir is already the folder containing images.
    sample_folder_dir = args.sample_dir
    create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=str, default="output/samples/inference")
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    args = parser.parse_args()
    main(args)
