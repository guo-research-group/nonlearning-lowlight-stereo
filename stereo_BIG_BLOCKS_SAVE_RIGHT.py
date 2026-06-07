import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from types import SimpleNamespace
from field_of_junctions import FieldOfJunctions
import torch.nn as nn
import torch
import os
from tqdm import tqdm

def read_pfm(file_path):
    """Return pfm with correct scaling"""
    data = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if data is None:
        raise ValueError(f"Failed to read: {file_path}")
    
    # Get scale factor from header
    with open(file_path, 'rb') as f:
        f.readline()  # type
        f.readline()  # dims
        scale = abs(float(f.readline().decode('utf-8').strip()))
    
    return data * scale


class StereoBigProcessor:
    """Process large stereo images using block based method"""
    
    def __init__(self, left_img_path, right_img_path, opts, gt_disparity_path=None):
        self.opts = opts
        self.boundary_threshold = 0.5
        
        left_img = cv2.imread(left_img_path)
        right_img = cv2.imread(right_img_path)

        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

        self.left_img_vis = left_img.copy()
        self.right_img_vis = right_img.copy()
        
        assert left_img.shape == right_img.shape
        
        left_img = left_img.astype(np.float32) / 255.0
        right_img = right_img.astype(np.float32) / 255.0
        
        self.left_img = left_img
        self.right_img = right_img
        self.H, self.W, self.C = left_img.shape
        
        self.gt_disparity_path = gt_disparity_path

    
    def process_big_stereo(self, verbose=True):
        """
        Block based processing for large images
        """
        
        opts = self.opts
        R = opts.R
        stride = opts.stride
        n_margin = opts.n_margin_patch
        
        # total number of foj patches that cover the image, ceil to ensure edge coverage
        H_patches = int(np.ceil((self.H - R) / stride)) + 1
        W_patches = int(np.ceil((self.W - R) / stride)) + 1
        
        if verbose:
            print(f"Image size: {self.H} x {self.W}")
            print(f"Global Patch Grid: {H_patches} x {W_patches}")
        
        # results tensors
        full_disp = torch.zeros(1, R, R, H_patches, W_patches)
        full_disp_right = torch.zeros(1, R, R, H_patches, W_patches)
        full_bound = torch.zeros(1, R, R, H_patches, W_patches)
        full_right_bound = torch.zeros(1, R, R, H_patches, W_patches)
        full_smooth = torch.zeros(3, R, R, H_patches, W_patches)
        full_right_smooth = torch.zeros(3, R, R, H_patches, W_patches)
        
        # pass 1 results tensors
        full_disp_p1 = torch.zeros(1, R, R, H_patches, W_patches)
        full_bound_p1 = torch.zeros(1, R, R, H_patches, W_patches)
        full_right_bound_p1 = torch.zeros(1, R, R, H_patches, W_patches)
        full_smooth_p1 = torch.zeros(3, R, R, H_patches, W_patches)
        full_right_smooth_p1 = torch.zeros(3, R, R, H_patches, W_patches)
        
        # get stride of chunks in terms of number of patches
        # so we need chunk_h_target number of patches needed to cover the block
        chunk_h_target = max(1, (opts.block_size[0] - R) // stride)
        chunk_w_target = max(1, (opts.block_size[1] - R) // stride)
        
        # iterate over the grid of chunks
        # (y, x) row, col of target output chunk in global patch grid
        for y in tqdm(list(range(0, H_patches, chunk_h_target)), desc="Processing rows"):
            for x in range(0, W_patches, chunk_w_target):
                
                # want to fill patches [y:y_end, x:x_end]
                # min to handle edge patches
                # this is the target core patches after margin patches are removed
                y_end = min(y + chunk_h_target, H_patches)
                x_end = min(x + chunk_w_target, W_patches)
                
                h_chunk = y_end - y
                w_chunk = x_end - x
                
                if verbose:
                    print(f"Processing chunk grid [{y}:{y_end}, {x}:{x_end}]")

                # to get the core patches [y:y_end] we need context [y-margin: y_end+margin]
                # so cy,cx specify the block fed into foj
                cy_start = max(0, y - n_margin)
                cy_end   = min(H_patches, y_end + n_margin)
                cx_start = max(0, x - n_margin)
                cx_end   = min(W_patches, x_end + n_margin)
                
                # calculate final block pixel coords
                px_start_y = cy_start * stride
                px_start_x = cx_start * stride
                
                # python slicing exclusive for end index so we need -1
                px_end_y = (cy_end - 1) * stride + R
                px_end_x = (cx_end - 1) * stride + R
                
                # PAD if needed - for bottom right blocks, px_end_y/x may be greater than H,W so we just pad
                pad_h = max(0, px_end_y - self.H)
                pad_w = max(0, px_end_x - self.W)
                
                # crop clamped from img, then pad as needed
                crop_y2 = min(px_end_y, self.H)
                crop_x2 = min(px_end_x, self.W)
                
                left_crop = self.left_img[px_start_y:crop_y2, px_start_x:crop_x2]
                right_crop = self.right_img[px_start_y:crop_y2, px_start_x:crop_x2]
                
                if pad_h > 0 or pad_w > 0:
                    left_crop = np.pad(left_crop, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
                    right_crop = np.pad(right_crop, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
                
                # RUN FOJ
                foj = FieldOfJunctions(left_crop, opts, right_img=right_crop)
                foj.optimize()
                
                # remove n_margin_patches / extract core patches
                # [cy_start:cy_end] -> subset [y:y_end]
                # local index of y is (y - cy_start)
                res_disp = foj.get_disparity_patches()      # [1, R, R, H_crop', W_crop']
                res_disp_right = foj.get_right_disparity_patches()
                res_bound = foj.get_boundary_patches()
                res_right_bound = foj.get_right_boundary_patches()
                res_smooth = foj.get_smoothed_patches()
                res_right_smooth = foj.get_right_smoothed_patches()
                
                local_y_start = y - cy_start
                local_y_end = local_y_start + h_chunk
                local_x_start = x - cx_start
                local_x_end = local_x_start + w_chunk
                
                # paste core patches
                full_disp[..., y:y_end, x:x_end] = res_disp[..., local_y_start:local_y_end, local_x_start:local_x_end]
                full_disp_right[..., y:y_end, x:x_end] = res_disp_right[..., local_y_start:local_y_end, local_x_start:local_x_end]
                full_bound[..., y:y_end, x:x_end] = res_bound[..., local_y_start:local_y_end, local_x_start:local_x_end]
                full_right_bound[..., y:y_end, x:x_end] = res_right_bound[..., local_y_start:local_y_end, local_x_start:local_x_end]
                full_smooth[..., y:y_end, x:x_end] = res_smooth[..., local_y_start:local_y_end, local_x_start:local_x_end]
                full_right_smooth[..., y:y_end, x:x_end] = res_right_smooth[..., local_y_start:local_y_end, local_x_start:local_x_end]
                
                # capture pass 1
                res_disp_p1 = foj.get_pass1_disparity_patches()
                res_bound_p1 = foj.get_pass1_boundary_patches()
                res_right_bound_p1 = foj.get_pass1_right_boundary_patches()
                res_smooth_p1 = foj.get_pass1_smoothed_patches()
                res_right_smooth_p1 = foj.get_pass1_right_smoothed_patches()

                full_disp_p1[..., y:y_end, x:x_end] = res_disp_p1[..., local_y_start:local_y_end, local_x_start:local_x_end]
                full_bound_p1[..., y:y_end, x:x_end] = res_bound_p1[..., local_y_start:local_y_end, local_x_start:local_x_end]
                full_right_bound_p1[..., y:y_end, x:x_end] = res_right_bound_p1[..., local_y_start:local_y_end, local_x_start:local_x_end]
                full_smooth_p1[..., y:y_end, x:x_end] = res_smooth_p1[..., local_y_start:local_y_end, local_x_start:local_x_end]
                full_right_smooth_p1[..., y:y_end, x:x_end] = res_right_smooth_p1[..., local_y_start:local_y_end, local_x_start:local_x_end]
                
                # free gpu mem
                del foj, res_disp, res_bound, res_right_bound, res_smooth, res_right_smooth
                torch.cuda.empty_cache()

        # fold to big image
        H_covered = (H_patches - 1) * stride + R
        W_covered = (W_patches - 1) * stride + R

        num_patches = nn.Fold(output_size=(H_covered, W_covered), kernel_size=R, stride=stride)(
            torch.ones(1, R**2, H_patches * W_patches)
        ).view(H_covered, W_covered)
        
        disparity_map = (nn.Fold(output_size=(H_covered, W_covered), kernel_size=R, stride=stride)(
            full_disp.view(1, R**2, -1)).view(H_covered, W_covered) / num_patches).numpy()

        disparity_map_right = (nn.Fold(output_size=(H_covered, W_covered), kernel_size=R, stride=stride)(
            full_disp_right.view(1, R**2, -1)).view(H_covered, W_covered) / num_patches).numpy()
        
        boundaries = (nn.Fold(output_size=(H_covered, W_covered), kernel_size=R, stride=stride)(
            full_bound.view(1, R**2, -1)).view(H_covered, W_covered) / num_patches).numpy()
        
        smoothed_left = (nn.Fold(output_size=(H_covered, W_covered), kernel_size=R, stride=stride)(
            full_smooth.view(1, 3*R**2, -1)).view(3, H_covered, W_covered) / 
            num_patches.unsqueeze(0)).permute(1, 2, 0).numpy()
        
        right_boundaries = (nn.Fold(output_size=(H_covered, W_covered), kernel_size=R, stride=stride)(
            full_right_bound.view(1, R**2, -1)).view(H_covered, W_covered) / num_patches).numpy()
        
        smoothed_right = (nn.Fold(output_size=(H_covered, W_covered), kernel_size=R, stride=stride)(
            full_right_smooth.view(1, 3*R**2, -1)).view(3, H_covered, W_covered) / 
            num_patches.unsqueeze(0)).permute(1, 2, 0).numpy()
        
        # fold pass 1
        disparity_map_p1 = (nn.Fold(output_size=(H_covered, W_covered), kernel_size=R, stride=stride)(
            full_disp_p1.view(1, R**2, -1)).view(H_covered, W_covered) / num_patches).numpy()
        
        boundaries_p1 = (nn.Fold(output_size=(H_covered, W_covered), kernel_size=R, stride=stride)(
            full_bound_p1.view(1, R**2, -1)).view(H_covered, W_covered) / num_patches).numpy()
        
        right_boundaries_p1 = (nn.Fold(output_size=(H_covered, W_covered), kernel_size=R, stride=stride)(
            full_right_bound_p1.view(1, R**2, -1)).view(H_covered, W_covered) / num_patches).numpy()

        smoothed_left_p1 = (nn.Fold(output_size=(H_covered, W_covered), kernel_size=R, stride=stride)(
            full_smooth_p1.view(1, 3*R**2, -1)).view(3, H_covered, W_covered) / 
            num_patches.unsqueeze(0)).permute(1, 2, 0).numpy()
        
        smoothed_right_p1 = (nn.Fold(output_size=(H_covered, W_covered), kernel_size=R, stride=stride)(
            full_right_smooth_p1.view(1, 3*R**2, -1)).view(3, H_covered, W_covered) / 
            num_patches.unsqueeze(0)).permute(1, 2, 0).numpy()
        
        if verbose:
            print(f"Disparity range: [{disparity_map.min():.3f}, {disparity_map.max():.3f}]")
        
        # crop to original size to remove padding added for edge blocks
        disparity_map = disparity_map[:self.H, :self.W]
        disparity_map_right = disparity_map_right[:self.H, :self.W]
        boundaries = boundaries[:self.H, :self.W]
        right_boundaries = right_boundaries[:self.H, :self.W]
        smoothed_left = smoothed_left[:self.H, :self.W]
        smoothed_right = smoothed_right[:self.H, :self.W]

        disparity_map_p1 = disparity_map_p1[:self.H, :self.W]
        boundaries_p1 = boundaries_p1[:self.H, :self.W]
        right_boundaries_p1 = right_boundaries_p1[:self.H, :self.W]
        smoothed_left_p1 = smoothed_left_p1[:self.H, :self.W]
        smoothed_right_p1 = smoothed_right_p1[:self.H, :self.W]

        return disparity_map, boundaries, smoothed_left, right_boundaries, smoothed_right, \
               disparity_map_p1, boundaries_p1, smoothed_left_p1, right_boundaries_p1, smoothed_right_p1, \
               disparity_map_right
    
def get_LRC_mask(disparity_L, disparity_R, threshold=1):
    """
    Returns left right consistency occlusion mask
    1 MEANS VALID
    """
    assert(disparity_L.shape == disparity_R.shape)
    H, W = disparity_L.shape

    disparity_L = torch.from_numpy(disparity_L)
    disparity_R = torch.from_numpy(disparity_R)

    x_coords = torch.arange(W)
    y_coords = torch.arange(H)
    X, Y = torch.meshgrid(x_coords, y_coords, indexing="xy")

    # make mapping from x to x-d
    target_x = torch.clip(X - disparity_L, 0, W - 1).long()
    # get right disparity at x-d
    right_disp_x = disparity_R[Y, target_x]
    occlusion_mask = torch.abs(disparity_L - right_disp_x) < threshold
    assert(occlusion_mask.shape == (H, W))

    #print(f"Percent valid: {occlusion_mask.float().mean():.2%}")
    
    return occlusion_mask.cpu().numpy()

def create_opts(num_init_iters=30, num_refine_iters=500, patch_size=21, stride=2, lambda_stereo_final=0.1, sgm_P1=20, sgm_P2=200, sgm_alpha_boundary=2.0):
    opts = SimpleNamespace()

    opts.R                        = patch_size
    opts.stride                   = stride
    opts.eta                      = 0.01
    opts.delta                    = 0.05
    opts.lr_angles                = 0.003
    opts.lr_x0y0                  = 0.03
    opts.lambda_boundary_final    = 0.5
    opts.lambda_color_final       = 0.1
    opts.nvals                    = 31
    opts.num_initialization_iters = num_init_iters
    opts.num_refinement_iters = num_refine_iters
    opts.greedy_step_every_iters  = 50
    opts.parallel_mode            = True
    opts.lr_disparity             = 0.15
    opts.lambda_stereo_final      = lambda_stereo_final

    opts.lambda_gcc = 0
    opts.lambda_stereo_raw = 0
    opts.lambda_stereo_recon = 0
    opts.search_lambda_stereo_recon = 0
    opts.lambda_soft_y = 0
    opts.lambda_soft_angle = 0
    opts.lambda_stereo_boundary = 0
    opts.stereo_delta = 0
    opts.STEREO_ANNEALING = False
    opts.occlusion_threshold = 1
    opts.alternating_optimization = False

    opts.sgm_alpha_boundary = sgm_alpha_boundary
    opts.max_disparity_pixels = 128 
    opts.disparity_step = 1

    opts.sgm_P1 = sgm_P1
    opts.sgm_P2 = sgm_P2

    opts.gt_disparity_L = None
    opts.gt_disparity_R = None

    return opts


def create_big_opts(block_size=(300, 300), n_margin_patch=4, num_init_iters=30, 
                    num_refine_iters=1000, patch_size=60, stride=8, lambda_stereo_final=0.1, sgm_P1=20, sgm_P2=200, sgm_alpha_boundary=2.0):
    """   
    Args:
        block_size: (height, width) of each block
        n_margin_patch: num edge patches to remove
    """
    opts = create_opts(num_init_iters, num_refine_iters, patch_size, stride, lambda_stereo_final, sgm_P1, sgm_P2, sgm_alpha_boundary)
    opts.block_size = block_size
    opts.n_margin_patch = n_margin_patch
    return opts