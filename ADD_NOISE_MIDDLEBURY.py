"""
Generates noisy middlebury dataset
"""

import cv2
import numpy as np
import os
from tqdm import tqdm


def add_poisson_gaussian_noise(img, alpha):
    sigma = 2
    img = img.astype(np.float64)
    
    img_prime = img / 255.0 * alpha
    
    img_noisy = np.random.poisson(img_prime).astype(np.float64) + sigma * np.random.randn(*img_prime.shape)
    img_noisy = img_noisy.clip(0, alpha).round()
    img_noisy = img_noisy / alpha * 255.0
    
    return img_noisy.astype(np.uint8)


def process_middlebury_pair(pair_dir, output_pair_dir, scale_factor=0.2, alpha=2):
    """
    Process a single pair - scale and add noise.
    """
    os.makedirs(output_pair_dir, exist_ok=True)
    
    left_path = os.path.join(pair_dir, "im0.png")
    if os.path.exists(left_path):
        img = cv2.imread(left_path)
        img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        img_noisy = add_poisson_gaussian_noise(img, alpha=alpha)
        cv2.imwrite(os.path.join(output_pair_dir, "im0.png"), img_noisy)
    
    right_path = os.path.join(pair_dir, "im1.png")
    if os.path.exists(right_path):
        img = cv2.imread(right_path)
        img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        img_noisy = add_poisson_gaussian_noise(img, alpha=alpha)
        cv2.imwrite(os.path.join(output_pair_dir, "im1.png"), img_noisy)



def main():
    INPUT_DIR = "middlebury2014"
    OUTPUT_DIR = "middlebury2014noisy_0.14scale"
    SCALE_FACTOR = 0.14
    ALPHA = 2
    
    # get all pairs
    pairs = sorted([d for d in os.listdir(INPUT_DIR) 
                    if os.path.isdir(os.path.join(INPUT_DIR, d))])
    
    print(f"Found {len(pairs)} Middlebury stereo pairs")
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Scale factor: {SCALE_FACTOR}")
    print(f"Noise alpha: {ALPHA}")
    print()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for pair_name in tqdm(pairs, desc="Processing pairs"):
        input_pair_dir = os.path.join(INPUT_DIR, pair_name)
        output_pair_dir = os.path.join(OUTPUT_DIR, pair_name)
        
        process_middlebury_pair(
            input_pair_dir, output_pair_dir,
            scale_factor=SCALE_FACTOR, alpha=ALPHA
        )
    
    print(f"\nDone! Noisy dataset saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
