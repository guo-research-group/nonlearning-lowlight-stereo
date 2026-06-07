import os
import cv2
import numpy as np

SCALE = 0.5

def add_poisson_gaussian_noise(img, alpha):
    sigma = 2
    img = img.astype(np.float64)
    
    img_prime = img / 255.0 * alpha
    
    img_noisy = np.random.poisson(img_prime).astype(np.float64) + sigma * np.random.randn(*img_prime.shape)
    img_noisy = img_noisy.clip(0, alpha).round()
    img_noisy = img_noisy / alpha * 255.0
    
    return img_noisy.astype(np.uint8)


def get_all_instereo_test_paths_no_noise(base_dir):
    dirs = os.listdir(base_dir)
    for dir in dirs:
        im0_path = os.path.join(base_dir, f'{dir}/left.png')
        im1_path = os.path.join(base_dir, f'{dir}/right.png')
        disp0_path = os.path.join(base_dir, f'{dir}/left_disp.png')
        if not os.path.exists(im0_path) or not os.path.exists(im1_path) or not os.path.exists(disp0_path):
            continue

        yield im0_path, im1_path, disp0_path

def add_noise(img_path, output_path, alpha):
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    img_noisy = add_poisson_gaussian_noise(img, alpha=alpha)
    cv2.imwrite(output_path, img_noisy)

base_output_dir = "instereo_test_set_noisy"

for path in get_all_instereo_test_paths_no_noise("instereo_test_set"):
    im0_path, im1_path, disp0_path = path
    scene_name = os.path.basename(os.path.dirname(im0_path))
    scene_output_dir = os.path.join(base_output_dir, scene_name)
    os.makedirs(scene_output_dir, exist_ok=True)
    
    add_noise(im0_path, os.path.join(scene_output_dir, "0_L.png"), alpha=2)
    add_noise(im1_path, os.path.join(scene_output_dir, "0_R.png"), alpha=2)

    gt_disparity = cv2.imread(disp0_path, cv2.IMREAD_UNCHANGED)
    gt_disparity = gt_disparity * SCALE / 100
    gt_disparity = cv2.resize(gt_disparity, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    
    np.save(os.path.join(scene_output_dir, "0_disp.npy"), gt_disparity)
