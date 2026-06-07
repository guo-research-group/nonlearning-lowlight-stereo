import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

if torch.cuda.is_available():
    dev = torch.device('cuda')
else:
    dev = torch.device('cpu')

class FieldOfJunctions:
    def __init__(self, img, opts, right_img):
        """
        Inputs
        ------
        img          Left image: a numpy array of shape [H, W, C]
        right_img    Right image: a numpy array of shape [H, W, C]
        opts   Object with the following attributes:
               R                          Patch size
               stride                     Stride for junctions (e.g. opts.stride == 1 is a dense field of junctions)
               eta                        Width parameter for Heaviside functions
               delta                      Width parameter for boundary maps
               lr_angles                  Angle learning rate
               lr_x0y0                    Vertex position learning rate
               lambda_boundary_final      Final value of spatial boundary consistency term
               lambda_color_final         Final value of spatial color consistency term
               nvals                      Number of values to query in Algorithm 2 from the paper
               num_initialization_iters   Number of initialization iterations
               num_refinement_iters       Number of refinement iterations
               greedy_step_every_iters    Frequency of "greedy" iteration (applying Algorithm 2 with consistency)
               parallel_mode              Whether or not to run Algorithm 2 in parallel over all `nvals` values.

               lambda_stereo_final        Final value of stereo consistency term
               lr_disparity               Disparity map lr
        """

        # Get image dimensions
        self.H, self.W, self.C = img.shape
        self.H_right, self.W_right, self.C_right = right_img.shape

        assert self.H == self.H_right and self.W == self.W_right and self.C == self.C_right, "L, R must have same shape"

        # Make sure number of patches in both dimensions is an integer
        assert (self.H - opts.R) % opts.stride == 0 and (self.W - opts.R) % opts.stride == 0, \
                "Number of patches must be an integer."

        # Number of patches (throughout the documentation H_patches and W_patches are denoted by H' and W' resp.)
        self.H_patches = (self.H - opts.R) // opts.stride + 1
        self.W_patches = (self.W - opts.R) // opts.stride + 1

        # Store total number of iterations (initialization + refinement)
        self.num_iters = opts.num_initialization_iters + opts.num_refinement_iters

        # Split image into overlapping patches, creating a tensor of shape [N, C, R, R, H', W']
        t_img = torch.tensor(img, device=dev).permute(2, 0, 1).unsqueeze(0)   # input image, shape [1, C, H, W]
        self.img_patches = nn.Unfold(opts.R, stride=opts.stride)(t_img).view(1, self.C, opts.R, opts.R,
                                                                             self.H_patches, self.W_patches)

        self.t_left_img = t_img.clone().float()

        # Needed for right img reconstruction loss
        t_right_img = torch.tensor(right_img, device=dev).permute(2, 0, 1).unsqueeze(0)
        self.right_img_patches = nn.Unfold(opts.R, stride=opts.stride)(t_right_img).view(1, self.C, opts.R, opts.R,
                                                                                         self.H_patches, self.W_patches)
        # Store full right image tensor for cost volume patch sampling
        self.t_right_img = t_right_img.float()  # [1, C, H, W]

        # Create pytorch variables for angles and vertex position for each patch - LEFT IMAGE
        self.angles = torch.zeros(1, 3, self.H_patches, self.W_patches, dtype=torch.float32, device=dev)
        self.x0y0   = torch.zeros(1, 2, self.H_patches, self.W_patches, dtype=torch.float32, device=dev)

        # Compute gradients for angles and vertex positions
        self.angles.requires_grad = True
        self.x0y0.requires_grad   = True

        # RIGHT image parameters - will be init later using left params - disparity
        self.right_angles = torch.zeros(1, 3, self.H_patches, self.W_patches, dtype=torch.float32, device=dev)
        self.right_x0y0   = torch.zeros(1, 2, self.H_patches, self.W_patches, dtype=torch.float32, device=dev)
        self.right_angles.requires_grad = True
        self.right_x0y0.requires_grad   = True

        # Compute number of patches containing each pixel: has shape [H, W]
        self.num_patches = torch.nn.Fold(output_size=[self.H, self.W],
                                         kernel_size=opts.R,
                                         stride=opts.stride)(torch.ones(1, opts.R**2,
                                                                        self.H_patches * self.W_patches,
                                                                        device=dev)).view(self.H, self.W)

        # Create local grid within each patch
        y, x = torch.meshgrid([torch.linspace(-1.0, 1.0, opts.R, device=dev),
                               torch.linspace(-1.0, 1.0, opts.R, device=dev)])
        self.x = x.view(1, opts.R, opts.R, 1, 1)
        self.y = y.view(1, opts.R, opts.R, 1, 1)

        # Optimization parameters
        adam_beta1 = 0.5
        adam_beta2 = 0.99
        adam_eps   = 1e-08

        # Create optimizers for angles and vertices - LEFT
        optimizer_angles = optim.Adam([self.angles],
                                       opts.lr_angles, [adam_beta1, adam_beta2], eps=adam_eps)
        optimizer_x0y0   = optim.Adam([self.x0y0],
                                       opts.lr_x0y0,   [adam_beta1, adam_beta2], eps=adam_eps)
        self.optimizers = [optimizer_angles, optimizer_x0y0]

        # optimizers for RIGHT params
        optimizer_right_angles = optim.Adam([self.right_angles],
                                             opts.lr_angles, [adam_beta1, adam_beta2], eps=adam_eps)
        optimizer_right_x0y0   = optim.Adam([self.right_x0y0],
                                             opts.lr_x0y0,   [adam_beta1, adam_beta2], eps=adam_eps)
        self.optimizers.extend([optimizer_right_angles, optimizer_right_x0y0])
        

        # Add disparity optimizer - LEFT
        # one horizontal disparity scalar for each junction / patch
        self.disparity = torch.zeros(1, 1, self.H_patches, self.W_patches, dtype=torch.float32, device=dev)
        self.disparity.requires_grad = True
        optimizer_disparity = optim.Adam([self.disparity],
                                         opts.lr_disparity, [adam_beta1, adam_beta2], eps=adam_eps)
        self.optimizers.append(optimizer_disparity)

        self.disparity_R = torch.zeros(1, 1, self.H_patches, self.W_patches, dtype=torch.float32, device=dev)

        # Values to search over in Algorithm 2: [0, 2pi) for angles, [-3, 3] for vertex position.
        self.angle_range = torch.linspace(0.0, 2*np.pi, opts.nvals+1, device=dev)[:opts.nvals]
        self.x0y0_range  = torch.linspace(-3.0, 3.0, opts.nvals, device=dev)

        # Save current global image and boundary map (initially None)
        self.global_image      = None
        self.global_boundaries = None

        self.global_right_image = None
        self.global_right_boundaries = None
        
        # Save opts
        self.opts = opts

    def optimize(self):
        """
        Optimize field of junctions.
        """
        self.optimize_with_cost_volume()
        
    def get_disparity_map(self):
        """
        Returns final disparity map of size (1, H, W)
        """
        # self.disparity has shape (1, 1, H', W') and we want to use local2global
        # which expects [N, C, R, R, H', W']
        disparity_patches = self.disparity.unsqueeze(-3).unsqueeze(-3).expand(
            -1, -1, self.opts.R, self.opts.R, -1, -1)
            
        disparity_map = self.local2global(disparity_patches)
        disparity_map = disparity_map[0, 0].detach().cpu().numpy()
        return disparity_map

    def get_disparity_patches(self):
        """
        Returns disparity as patches of shape [1, R, R, H', W'] for block-based processing.
        """
        # self.disparity has shape (1, 1, H', W')
        disparity_patches = self.disparity.unsqueeze(-3).unsqueeze(-3).expand(
            -1, -1, self.opts.R, self.opts.R, -1, -1)
        return disparity_patches[0].detach().clone()  # [1, R, R, H', W']

    def get_right_disparity_patches(self):
        """
        Returns right disparity as patches of shape [1, R, R, H', W'] for block-based processing.
        """
        disparity_patches = self.disparity_R.unsqueeze(-3).unsqueeze(-3).expand(
            -1, -1, self.opts.R, self.opts.R, -1, -1)
        return disparity_patches[0].detach().clone()
    
    
    def get_boundary_patches(self):
        """
        Returns boundary map as patches of shape [1, R, R, H', W'] for block-based processing.
        """
        # global_boundaries has shape [1, 1, H, W], unfold to patches
        boundary_patches = nn.Unfold(self.opts.R, stride=self.opts.stride)(
            self.global_boundaries.detach()).view(1, 1, self.opts.R, self.opts.R, self.H_patches, self.W_patches)
        return boundary_patches[0].clone()  # [1, R, R, H', W']

    def get_right_boundary_patches(self):
        """
        Returns right boundary map as patches of shape [1, R, R, H', W'] for block-based processing.
        """
        # global_boundaries has shape [1, 1, H, W], unfold to patches
        boundary_patches = nn.Unfold(self.opts.R, stride=self.opts.stride)(
            self.global_right_boundaries.detach()).view(1, 1, self.opts.R, self.opts.R, self.H_patches, self.W_patches)
        return boundary_patches[0].clone()  # [1, R, R, H', W']

    def get_smoothed_patches(self):
        """
        Returns smoothed image as patches of shape [C, R, R, H', W'] for block-based processing.
        """
        # global_image has shape [1, C, H, W], unfold to patches
        smooth_patches = nn.Unfold(self.opts.R, stride=self.opts.stride)(
            self.global_image.detach()).view(1, self.C, self.opts.R, self.opts.R, self.H_patches, self.W_patches)
        return smooth_patches[0].clone()  # [C, R, R, H', W']

    def get_right_smoothed_patches(self):
        """
        Returns smoothed right image as patches of shape [C, R, R, H', W'] for block-based processing.
        """
        # global_right_image has shape [1, C, H, W], unfold to patches
        smooth_patches = nn.Unfold(self.opts.R, stride=self.opts.stride)(
            self.global_right_image.detach()).view(1, self.C, self.opts.R, self.opts.R, self.H_patches, self.W_patches)
        return smooth_patches[0].clone()  # [C, R, R, H', W']

    def get_pass1_disparity_patches(self):
        """
        Returns pass 1 disparity as patches of shape [1, R, R, H', W'] for block-based processing.
        """
        d_tensor = torch.as_tensor(self.pass1_disparity_map, device=dev, dtype=torch.float32).view(1, 1, self.H_patches, self.W_patches)
        disparity_patches = d_tensor.unsqueeze(-3).unsqueeze(-3).expand(-1, -1, self.opts.R, self.opts.R, -1, -1)
        return disparity_patches[0].detach().clone()

    def get_pass1_boundary_patches(self):
        """
        Returns pass 1 boundary map as patches of shape [1, R, R, H', W'] for block-based processing.
        """
        boundary_patches = nn.Unfold(self.opts.R, stride=self.opts.stride)(
            self.pass1_global_boundaries.detach()).view(1, 1, self.opts.R, self.opts.R, self.H_patches, self.W_patches)
        return boundary_patches[0].clone()

    def get_pass1_right_boundary_patches(self):
        """
        Returns pass 1 right boundary map as patches of shape [1, R, R, H', W'] for block-based processing.
        """
        boundary_patches = nn.Unfold(self.opts.R, stride=self.opts.stride)(
            self.pass1_global_right_boundaries.detach()).view(1, 1, self.opts.R, self.opts.R, self.H_patches, self.W_patches)
        return boundary_patches[0].clone()

    def get_pass1_smoothed_patches(self):
        """
        Returns pass 1 smoothed image as patches of shape [C, R, R, H', W'] for block-based processing.
        """
        smooth_patches = nn.Unfold(self.opts.R, stride=self.opts.stride)(
            self.pass1_global_image.detach()).view(1, self.C, self.opts.R, self.opts.R, self.H_patches, self.W_patches)
        return smooth_patches[0].clone()

    def get_pass1_right_smoothed_patches(self):
        """
        Returns pass 1 smoothed right image as patches of shape [C, R, R, H', W'] for block-based processing.
        """
        smooth_patches = nn.Unfold(self.opts.R, stride=self.opts.stride)(
            self.pass1_global_right_image.detach()).view(1, self.C, self.opts.R, self.opts.R, self.H_patches, self.W_patches)
        return smooth_patches[0].clone()

    def get_right_loss(self, dists, colors, patches, lmbda_boundary, lmbda_color, disable_multi_pass=True, params=None):
        """
        Compute the objective of our model (see Equation 8 of the paper).

        Inputs
        ------
        dists             Tensor of shape [N, 2, R, R, H', W'] with samples of the two distance functions for every patch
        colors            Tensor of shape [N, C, 3, H', W'] storing the C colors at each patch
        patches           Tensor of shape [N, C, R, R, H', W'] with each patch having color c_i^{(j)} at the jth wedge, for each i
        lmbda_boundary    Spatial consistency boundary loss weight
        lmbda_color       Spatial consistency color loss weight

        Outputs
        -------
                 Tensor of shape [N, H', W'] with the loss at each patch
        """
        # Compute negative log-likelihood for each patch (shape [N, H', W'])
        loss_per_patch = ((self.right_img_patches - patches) ** 2).mean(-3).mean(-3).sum(1)

        # Add spatial consistency loss for each patch, if lambda > 0
        if lmbda_boundary > 0.0:
            loss_per_patch = loss_per_patch + lmbda_boundary * self.get_right_boundary_consistency_term(dists)

        if lmbda_color > 0.0:
            loss_per_patch = loss_per_patch + lmbda_color * self.get_right_color_consistency_term(dists, colors)

        return loss_per_patch

    def get_right_boundary_consistency_term(self, dists):
        """
        Compute the spatial consistency term.

        Inputs
        ------
        dists    Tensor of shape [N, 2, R, R, H', W'] with samples of the two distance functions for every patch

        Outputs
        -------
                 Tensor of shape [N, H', W'] with the consistency loss at each patch
        """
        # Split global boundaries into patches
        curr_global_boundaries_patches = nn.Unfold(self.opts.R, stride=self.opts.stride)(
            self.global_right_boundaries.detach()).view(1, 1, self.opts.R,self.opts.R, self.H_patches, self.W_patches)

        # Get local boundaries defined using the queried parameters (defined by `dists`)
        local_boundaries = self.dists2boundaries(dists)

        # Compute consistency term
        consistency = ((local_boundaries - curr_global_boundaries_patches) ** 2).mean(2).mean(2)

        return consistency[:, 0, :, :]

    def get_right_color_consistency_term(self, dists, colors):
        """
        Compute the spatial consistency term.

        Inputs
        ------
        dists    Tensor of shape [N, 2, R, R, H', W'] with samples of the two distance functions for every patch

        Outputs
        -------
                 Tensor of shape [N, H', W'] with the consistency loss at each patch
        """
        # Split into patches
        curr_global_image_patches = nn.Unfold(self.opts.R, stride=self.opts.stride)(
            self.global_right_image.detach()).view(1, self.C, self.opts.R,self.opts.R, self.H_patches, self.W_patches)

        wedges = self.dists2indicators(dists)  # shape [N, 3, R, R, H', W']

        # Compute consistency term
        consistency = (wedges.unsqueeze(1) * (
            colors.unsqueeze(-3).unsqueeze(-3) - curr_global_image_patches.unsqueeze(2)) ** 2).mean(-3).mean(-3).sum(1).sum(1)

        return consistency

    def get_right_dists_and_patches(self, params, lmbda_color=0.0, sampled_left_img_patches=None):
        """
        Compute distance functions and piecewise-constant patches given junction parameters.

        Inputs
        ------
        params   Tensor of shape [N, 5, H', W'] holding N field of junctions parameters. Each
                 5-vector has format (angle1, angle2, angle3, x0, y0).

        Outputs
        -------
        dists    Tensor of shape [N, 2, R, R, H', W'] with samples of the two distance functions for every patch
        colors   Tensor of shape [N, C, 3, H', W']
        patches  Tensor of shape [N, C, R, R, H', W'] with the constant color function at each of the 3 wedges
        """

        # Get dists
        dists = self.params2dists(params)    # shape [N, 2, R, R, H', W']

        # Get wedge indicator functions
        wedges = self.dists2indicators(dists)  # shape [N, 3, R, R, H', W']

        if lmbda_color >= 0 and self.global_right_image is not None:
            curr_global_image_patches = nn.Unfold(self.opts.R, stride=self.opts.stride)(
                self.global_right_image.detach()).view(1, self.C, self.opts.R,self.opts.R, self.H_patches, self.W_patches)
            
            # IF pass 2 and patches are not occluded, we calculate color using both left
            # patch at (x,y) and right patch at (x-d,y)
            if sampled_left_img_patches is None:
                combined_patches = self.right_img_patches.clone()
            else:
                mask = sampled_left_img_patches.sum(dim=(1,2,3), keepdim=True) > 0
                combined_patches = torch.where(mask, (self.right_img_patches + sampled_left_img_patches) / 2.0, self.right_img_patches)

            
            numerator = ((combined_patches + lmbda_color *
                          curr_global_image_patches).unsqueeze(2) * wedges.unsqueeze(1)).sum(-3).sum(-3)
            denominator = (1.0 + lmbda_color) * wedges.sum(-3).sum(-3).unsqueeze(1)
            
            colors = numerator / (denominator + 1e-10)
        else:
            # Get best color for each wedge and each patch
            colors = (self.right_img_patches.unsqueeze(2) * wedges.unsqueeze(1)).sum(-3).sum(-3) / \
                     (wedges.sum(-3).sum(-3).unsqueeze(1) + 1e-10)

        # Fill wedges with optimal colors
        patches = (wedges.unsqueeze(1) * colors.unsqueeze(-3).unsqueeze(-3)).sum(dim=2)

        return dists, colors, patches
            
    def get_loss(self, dists, colors, patches, lmbda_boundary, lmbda_color, disable_multi_pass=True, params=None):
        """
        Compute the objective of our model (see Equation 8 of the paper).

        Inputs
        ------
        dists             Tensor of shape [N, 2, R, R, H', W'] with samples of the two distance functions for every patch
        colors            Tensor of shape [N, C, 3, H', W'] storing the C colors at each patch
        patches           Tensor of shape [N, C, R, R, H', W'] with each patch having color c_i^{(j)} at the jth wedge, for each i
        lmbda_boundary    Spatial consistency boundary loss weight
        lmbda_color       Spatial consistency color loss weight

        Outputs
        -------
                 Tensor of shape [N, H', W'] with the loss at each patch
        """
        # Compute negative log-likelihood for each patch (shape [N, H', W'])
        loss_per_patch = ((self.img_patches - patches) ** 2).mean(-3).mean(-3).sum(1)

        # Add spatial consistency loss for each patch, if lambda > 0
        if lmbda_boundary > 0.0:
            loss_per_patch = loss_per_patch + lmbda_boundary * self.get_boundary_consistency_term(dists)

        if lmbda_color > 0.0:
            loss_per_patch = loss_per_patch + lmbda_color * self.get_color_consistency_term(dists, colors)

        return loss_per_patch

    def get_boundary_consistency_term(self, dists):
        """
        Compute the spatial consistency term.

        Inputs
        ------
        dists    Tensor of shape [N, 2, R, R, H', W'] with samples of the two distance functions for every patch

        Outputs
        -------
                 Tensor of shape [N, H', W'] with the consistency loss at each patch
        """
        # Split global boundaries into patches
        curr_global_boundaries_patches = nn.Unfold(self.opts.R, stride=self.opts.stride)(
            self.global_boundaries.detach()).view(1, 1, self.opts.R,self.opts.R, self.H_patches, self.W_patches)

        # Get local boundaries defined using the queried parameters (defined by `dists`)
        local_boundaries = self.dists2boundaries(dists)

        # Compute consistency term
        consistency = ((local_boundaries - curr_global_boundaries_patches) ** 2).mean(2).mean(2)

        return consistency[:, 0, :, :]

    def get_color_consistency_term(self, dists, colors):
        """
        Compute the spatial consistency term.

        Inputs
        ------
        dists    Tensor of shape [N, 2, R, R, H', W'] with samples of the two distance functions for every patch

        Outputs
        -------
                 Tensor of shape [N, H', W'] with the consistency loss at each patch
        """
        # Split into patches
        curr_global_image_patches = nn.Unfold(self.opts.R, stride=self.opts.stride)(
            self.global_image.detach()).view(1, self.C, self.opts.R,self.opts.R, self.H_patches, self.W_patches)

        wedges = self.dists2indicators(dists)  # shape [N, 3, R, R, H', W']

        # Compute consistency term
        consistency = (wedges.unsqueeze(1) * (
            colors.unsqueeze(-3).unsqueeze(-3) - curr_global_image_patches.unsqueeze(2)) ** 2).mean(-3).mean(-3).sum(1).sum(1)

        return consistency

    def get_dists_and_patches(self, params, lmbda_color=0.0, sampled_right_img_patches=None):
        """
        Compute distance functions and piecewise-constant patches given junction parameters.

        Inputs
        ------
        params   Tensor of shape [N, 5, H', W'] holding N field of junctions parameters. Each
                 5-vector has format (angle1, angle2, angle3, x0, y0).

        Outputs
        -------
        dists    Tensor of shape [N, 2, R, R, H', W'] with samples of the two distance functions for every patch
        colors   Tensor of shape [N, C, 3, H', W']
        patches  Tensor of shape [N, C, R, R, H', W'] with the constant color function at each of the 3 wedges
        """

        # Get dists
        dists = self.params2dists(params)    # shape [N, 2, R, R, H', W']

        # Get wedge indicator functions
        wedges = self.dists2indicators(dists)  # shape [N, 3, R, R, H', W']

        if lmbda_color >= 0 and self.global_image is not None:
            curr_global_image_patches = nn.Unfold(self.opts.R, stride=self.opts.stride)(
                self.global_image.detach()).view(1, self.C, self.opts.R,self.opts.R, self.H_patches, self.W_patches)
            
            # IF pass 2 and patches are not occluded, we calculate color using both left
            # patch at (x,y) and right patch at (x-d,y)
            if sampled_right_img_patches is None:
                combined_patches = self.img_patches.clone()
            else:
                mask = sampled_right_img_patches.sum(dim=(1,2,3), keepdim=True) > 0
                combined_patches = torch.where(mask, (self.img_patches + sampled_right_img_patches) / 2.0, self.img_patches)

            
            numerator = ((combined_patches + lmbda_color *
                          curr_global_image_patches).unsqueeze(2) * wedges.unsqueeze(1)).sum(-3).sum(-3)
            denominator = (1.0 + lmbda_color) * wedges.sum(-3).sum(-3).unsqueeze(1)
            
            colors = numerator / (denominator + 1e-10)
        else:
            # Get best color for each wedge and each patch
            colors = (self.img_patches.unsqueeze(2) * wedges.unsqueeze(1)).sum(-3).sum(-3) / \
                         (wedges.sum(-3).sum(-3).unsqueeze(1) + 1e-10)

        # Fill wedges with optimal colors
        patches = (wedges.unsqueeze(1) * colors.unsqueeze(-3).unsqueeze(-3)).sum(dim=2)

        return dists, colors, patches

    def dists2boundaries(self, dists):
        """
        Compute boundary map for each patch, given distance functions. The width of the boundary is determined
        by opts.delta.

        Inputs
        ------
        dists    Tensor of shape [N, 2, R, R, H', W'] with samples of the two distance functions for every patch

        Outputs
        -------
                 Tensor of shape [N, 1, R, R, H', W'] with values of boundary map for every patch
        """
        # Find places where either distance transform is small, except where d1 > 0 and d2 < 0
        d1 = dists[:, 0:1, :, :, :, :]
        d2 = dists[:, 1:2, :, :, :, :]
        minabsdist = torch.where(d1 < 0.0, -d1, torch.where(d2 < 0.0, torch.min(d1, -d2), torch.min(d1, d2)))

        return 1.0 / (1.0 + (minabsdist / self.opts.delta) ** 2)

    def local2global(self, patches):
        """
        Compute average value for each pixel over all patches containing it.
        For example, this can be used to compute the global boundary maps, or the boundary-aware smoothed image.

        Inputs
        ------
        patches   Tensor of shape [N, C, R, R, H', W']. patches[n, :, :, :, i, j] is an RxR C-channel patch
                  at the (i, j)th spatial position of the nth entry.


        Outputs
        -------
                  Tensor of shape [N, C, H, W] of averages over all patches containing each pixel.
        """
        N = patches.shape[0]
        C = patches.shape[1]
        return torch.nn.Fold(output_size=[self.H, self.W], kernel_size=self.opts.R, stride=self.opts.stride)(
            patches.view(N, C*self.opts.R**2, -1)).view(N, C, self.H, self.W) / \
                self.num_patches.unsqueeze(0).unsqueeze(0)

    def get_best_inds(self, params, lmbda_boundary, lmbda_color, disable_multi_pass=True):
        """
        Compute the best index along the 0th dimension of `params` for each pixel position.
        Has two possible modes determined by self.opts.parallel_mode:
        1) When True, all N values are computed in parallel (generally faster, requires more memory)
        2) When False, the values are computed sequentially (generally slower, requires less memory)

        Inputs
        ------
        params            Tensor of shape [N, 5, H', W'] holding N field of junctions parameters. Each
                          5-vector has format (angle1, angle2, angle3, x0, y0).
        lmbda_boundary    Spatial consistency boundary loss weight
        lmbda_color       Spatial consistency color loss weight

        Outputs
        -------
                          Tensor of shape [H', W'] with each value in {0, ..., N-1} holding the
                          index of the best junction parameters at that position.
        """
        if self.opts.parallel_mode:
            dists, colors, smooth_patches = self.get_dists_and_patches(params, lmbda_color)
            loss_per_patch = self.get_loss(dists, colors, smooth_patches, lmbda_boundary, lmbda_color, disable_multi_pass=disable_multi_pass, params=params)
            best_ind = loss_per_patch.argmin(dim=0)

        else:
            # First initialize tensors
            best_ind            = torch.zeros(self.H_patches, self.W_patches, device=dev, dtype=torch.int64)
            best_loss_per_patch = torch.zeros(self.H_patches, self.W_patches, device=dev) + 1e10

            # Now fill tensors by iterating over the junction dimension and choosing the best junction parameters
            for n in range(params.shape[0]):
                dists, colors, smooth_patches = self.get_dists_and_patches(params[n:n+1, :, :, :], lmbda_color)

                loss_per_patch = self.get_loss(dists, colors, smooth_patches, lmbda_boundary, lmbda_color, disable_multi_pass=disable_multi_pass, params=params[n:n+1, :, :, :])

                improved_inds       = loss_per_patch[0] < best_loss_per_patch
                best_ind            = torch.where(improved_inds, torch.tensor(n, device=dev, dtype=torch.int64), best_ind)
                best_loss_per_patch = torch.where(improved_inds, loss_per_patch, best_loss_per_patch)

        return best_ind

    def params2dists(self, params, tau=1e-1):
        """
        Compute distance functions from field of junctions.

        Inputs
        ------
        params   Tensor of shape [N, 5, H', W'] holding N field of junctions parameters. Each
                 5-vector has format (angle1, angle2, angle3, x0, y0).
        tau      Constant used for lifting the level set function to be either entirely positive
                 or entirely negative when an angle approaches 0 or 2pi.


        Outputs
        -------
                 Tensor of shape [N, 2, R, R, H', W'] with samples of the two distance functions for every patch
        """
        x0     = params[:, 3, :, :].unsqueeze(1).unsqueeze(1)   # shape [N, 1, 1, H', W']
        y0     = params[:, 4, :, :].unsqueeze(1).unsqueeze(1)   # shape [N, 1, 1, H', W']

        # Sort so angle1 <= angle2 <= angle3 (mod 2pi)
        angles = torch.remainder(params[:, :3, :, :], 2 * np.pi)
        angles = torch.sort(angles, dim=1)[0]

        angle1 = angles[:, 0, :, :].unsqueeze(1).unsqueeze(1)   # shape [N, 1, 1, H', W']
        angle2 = angles[:, 1, :, :].unsqueeze(1).unsqueeze(1)   # shape [N, 1, 1, H', W']
        angle3 = angles[:, 2, :, :].unsqueeze(1).unsqueeze(1)   # shape [N, 1, 1, H', W']

        # Define another angle halfway between angle3 and angle1, clockwise from angle3
        # This isn't critical but it seems a bit more stable for computing gradients
        angle4 = 0.5 * (angle1 + angle3) + \
                     torch.where(torch.remainder(0.5 * (angle1 - angle3), 2 * np.pi) >= np.pi,
                                 torch.ones_like(angle1) * np.pi, torch.zeros_like(angle1))

        def g(dtheta):
            # Map from [0, 2pi] to [-1, 1]
            return (dtheta / np.pi - 1.0) ** 35

        # Compute the two distance functions
        sgn42 = torch.where(torch.remainder(angle2 - angle4, 2 * np.pi) < np.pi,
                            torch.ones_like(angle2), -torch.ones_like(angle2))
        tau42 = g(torch.remainder(angle2 - angle4, 2*np.pi)) * tau

        dist42 = sgn42 * torch.min( sgn42 * (-torch.sin(angle4) * (self.x - x0) + torch.cos(angle4) * (self.y - y0)),
                                   -sgn42 * (-torch.sin(angle2) * (self.x - x0) + torch.cos(angle2) * (self.y - y0))) + tau42

        sgn13 = torch.where(torch.remainder(angle3 - angle1, 2 * np.pi) < np.pi,
                            torch.ones_like(angle3), -torch.ones_like(angle3))
        tau13 = g(torch.remainder(angle3 - angle1, 2*np.pi)) * tau
        dist13 = sgn13 * torch.min( sgn13 * (-torch.sin(angle1) * (self.x - x0) + torch.cos(angle1) * (self.y - y0)),
                                   -sgn13 * (-torch.sin(angle3) * (self.x - x0) + torch.cos(angle3) * (self.y - y0))) + tau13

        return torch.stack([dist13, dist42], dim=1)

    def dists2indicators(self, dists):
        """
        Computes the indicator functions u_1, u_2, u_3 from the distance functions d_{13}, d_{12}

        Inputs
        ------
        dists   Tensor of shape [N, 2, R, R, H', W'] with samples of the two distance functions for every patch

        Outputs
        -------
                Tensor of shape [N, 3, R, R, H', W'] with samples of the three indicator functions for every patch
        """
        # Apply smooth Heaviside function to distance functions
        hdists = 0.5 * (1.0 + (2.0 / np.pi) * torch.atan(dists / self.opts.eta))

        # Convert Heaviside functions into wedge indicator functions
        return torch.stack([1.0 - hdists[:, 0, :, :, :, :],
                                  hdists[:, 0, :, :, :, :] * (1.0 - hdists[:, 1, :, :, :, :]),
                                  hdists[:, 0, :, :, :, :] *        hdists[:, 1, :, :, :, :]], dim=1)

    # --------------------------------------- COST VOLUME METHODS ---------------------------------------

    def sample_right_image_at_disparity(self, disparity_pixels):
        """
        For patch grid position (x0, y0), samples the right image at position (x0 - d, y0)
        where d is the disparity in pixels
        
        Uses zero padding and returns a validity mask indicating which pixels have valid data
        for later masking
        
        Returns:
            right_patches_shifted: tensor of shape [1, C, R, R, H', W']
            valid_mask: tensor of shape [1, 1, R, R, H', W'] with 1 for valid pixels
        """
        R = self.opts.R
        stride = self.opts.stride
        d = int(disparity_pixels)
        
        if d == 0:
            right_shifted = self.t_right_img
            valid_mask_img = torch.ones_like(self.t_right_img)
        else:
            # pad left with zeros, crop right
            right_shifted = F.pad(self.t_right_img, (d, 0, 0, 0), mode='constant', value=0)
            right_shifted = right_shifted[:, :, :, :-d]
            
            # same for valid mask
            valid_mask_img = torch.ones_like(self.t_right_img)
            valid_mask_img = F.pad(valid_mask_img, (d, 0, 0, 0), mode='constant', value=0)
            valid_mask_img = valid_mask_img[:, :, :, :-d]
        
        right_patches_shifted = nn.Unfold(R, stride=stride)(right_shifted).view(
            1, self.C, R, R, self.H_patches, self.W_patches)
        
        valid_mask_patches = nn.Unfold(R, stride=stride)(valid_mask_img).view(
            1, self.C, R, R, self.H_patches, self.W_patches)
        
        return right_patches_shifted, valid_mask_patches

    def sample_left_image_at_disparity(self, disparity_pixels):
        """
        Same as sample_right_image_at_disparity but for left image
        """
        R = self.opts.R
        stride = self.opts.stride
        d = int(disparity_pixels)
        
        if d == 0:
            left_shifted = self.t_left_img
            valid_mask_img = torch.ones_like(self.t_left_img)
        else:
            left_shifted = F.pad(self.t_left_img, (0, d, 0, 0), mode='constant', value=0)
            left_shifted = left_shifted[:, :, :, d:]
            
            valid_mask_img = torch.ones_like(self.t_left_img)
            valid_mask_img = F.pad(valid_mask_img, (0, d, 0, 0), mode='constant', value=0)
            valid_mask_img = valid_mask_img[:, :, :, d:]
        
        left_patches_shifted = nn.Unfold(R, stride=stride)(left_shifted).view(
            1, self.C, R, R, self.H_patches, self.W_patches)
        
        valid_mask_patches = nn.Unfold(R, stride=stride)(valid_mask_img).view(
            1, self.C, R, R, self.H_patches, self.W_patches)
        
        return left_patches_shifted, valid_mask_patches

    def build_cost_volume(self):
        """
        Run after self.angles, self.x0y0, self.right_angles, self.right_x0y0 are optimized
        
        BIDIRECTIONAL SYMMETRIC COST VOLUME:
        For each patch position (x0, y0) and disparity d:
            COST = MSE(Reconstructed_Left(x0, y0), Raw_Right(x0-d, y0)) + 
                   MSE(Reconstructed_Right(x0, y0), Raw_Left(x0+d, y0)) +
                   MSE(Reconstructed_Left(x0, y0), Reconstructed_Right(x0+d, y0))
        
        Returns:
            cost_volume: np array of shape [H', W', D]
            disparity_candidates: np array of shape [D,] with disparity values in pixels
        """
        max_disp = getattr(self.opts, 'max_disparity_pixels', 128)
        step = getattr(self.opts, 'disparity_step', 1)
        
        disparity_candidates = np.arange(0, max_disp, step)
        
        # get reconstructed left patches
        left_params = torch.cat([self.angles, self.x0y0], dim=1).detach()
        left_dists = self.params2dists(left_params)
        left_wedges = self.dists2indicators(left_dists)  # [1, 3, R, R, H', W']
        
        left_colors = (self.img_patches.unsqueeze(2) * left_wedges.unsqueeze(1)).sum(-3).sum(-3) / \
                    (left_wedges.sum(-3).sum(-3).unsqueeze(1) + 1e-10)

        # [1, C, R, R, H', W']
        left_reconstruction = (left_wedges.unsqueeze(1) * left_colors.unsqueeze(-3).unsqueeze(-3)).sum(dim=2)
        
        # get reconstructed right patches
        right_params = torch.cat([self.right_angles, self.right_x0y0], dim=1).detach()
        right_dists = self.params2dists(right_params)
        right_wedges = self.dists2indicators(right_dists)  # [1, 3, R, R, H', W']
        
        right_colors = (self.right_img_patches.unsqueeze(2) * right_wedges.unsqueeze(1)).sum(-3).sum(-3) / \
                    (right_wedges.sum(-3).sum(-3).unsqueeze(1) + 1e-10)

        # [1, C, R, R, H', W']
        right_reconstruction = (right_wedges.unsqueeze(1) * right_colors.unsqueeze(-3).unsqueeze(-3)).sum(dim=2)
        
        # boundary maps for weighted p2
        R = self.opts.R
        stride = self.opts.stride
        left_boundaries = self.local2global(self.dists2boundaries(left_dists))  
        left_boundaries = nn.Unfold(R, stride=stride)(left_boundaries).view(
            1, 1, R, R, self.H_patches, self.W_patches)
        self.left_boundary_strength = left_boundaries.max(dim=2)[0].max(dim=2)[0][0, 0].cpu().numpy()  # [H', W']
        
        right_boundaries = self.local2global(self.dists2boundaries(right_dists))
        right_boundaries = nn.Unfold(R, stride=stride)(right_boundaries).view(
            1, 1, R, R, self.H_patches, self.W_patches)
        self.right_boundary_strength = right_boundaries.max(dim=2)[0].max(dim=2)[0][0, 0].cpu().numpy()  # [H', W']

        # Get global reconstructions for shifted comparison
        global_left_recon = self.local2global(left_reconstruction)  # [1, C, H, W]
        global_right_recon = self.local2global(right_reconstruction)  # [1, C, H, W]
        
        # compute cost volume
        D = len(disparity_candidates)
        
        # [H', W', D]
        cost_volume_left = np.zeros((self.H_patches, self.W_patches, D), dtype=np.float32)
        cost_volume_right = np.zeros((self.H_patches, self.W_patches, D), dtype=np.float32)
        with torch.no_grad():
            for i, d in tqdm(enumerate(disparity_candidates), desc="Building cost volume", total=D):
                right_patches_shifted, valid_mask_R = self.sample_right_image_at_disparity(d)
                left_patches_shifted, valid_mask_L = self.sample_left_image_at_disparity(d)

                # compute MSE only over valid pixels to avoid matching padding
                # [1, C, R, R, H', W']
                diff_L = (left_reconstruction - right_patches_shifted) ** 2
                diff_R = (right_reconstruction - left_patches_shifted) ** 2
                
                # sum over [C, R, R] dims (1, 2, 3)
                sum_diff_L = (diff_L * valid_mask_R).sum(dim=(1, 2, 3))  # [1, H', W']
                sum_mask_L = valid_mask_R.sum(dim=(1, 2, 3))  # [1, H', W']
                mse_L = sum_diff_L / (sum_mask_L + 1e-10)
                
                sum_diff_R = (diff_R * valid_mask_L).sum(dim=(1, 2, 3))
                sum_mask_R = valid_mask_L.sum(dim=(1, 2, 3))
                mse_R = sum_diff_R / (sum_mask_R + 1e-10)

                mse_L = mse_L[0].cpu().numpy()
                mse_R = mse_R[0].cpu().numpy()
                
                # reconstruction vs reconstruction cost
                d_int = int(d)
                if d_int == 0:
                    right_recon_shifted = global_right_recon
                    left_recon_shifted = global_left_recon
                else:
                    # pad left with zeros, crop right
                    right_recon_shifted = F.pad(global_right_recon, (d_int, 0, 0, 0), mode='constant', value=0)
                    right_recon_shifted = right_recon_shifted[:, :, :, :-d_int]
                    
                    # pad right with zeros, crop left
                    left_recon_shifted = F.pad(global_left_recon, (0, d_int, 0, 0), mode='constant', value=0)
                    left_recon_shifted = left_recon_shifted[:, :, :, d_int:]
                
                right_recon_patches = nn.Unfold(R, stride=stride)(right_recon_shifted).view(
                    1, self.C, R, R, self.H_patches, self.W_patches)
                left_recon_patches = nn.Unfold(R, stride=stride)(left_recon_shifted).view(
                    1, self.C, R, R, self.H_patches, self.W_patches)
                
                diff_recon_L = (left_reconstruction - right_recon_patches) ** 2
                diff_recon_R = (right_reconstruction - left_recon_patches) ** 2
                
                # symmetric raw vs recon costs - left loss += warped right image vs left image
                diff_sym_L = (right_recon_patches - self.img_patches) ** 2
                diff_sym_R = (left_recon_patches - self.right_img_patches) ** 2

                sum_diff_recon_L = (diff_recon_L * valid_mask_R).sum(dim=(1, 2, 3))
                sum_diff_recon_R = (diff_recon_R * valid_mask_L).sum(dim=(1, 2, 3))
                
                sum_diff_sym_L = (diff_sym_L * valid_mask_R).sum(dim=(1, 2, 3))
                sum_diff_sym_R = (diff_sym_R * valid_mask_L).sum(dim=(1, 2, 3))

                mse_recon_L = sum_diff_recon_L / (sum_mask_L + 1e-10)
                mse_recon_R = sum_diff_recon_R / (sum_mask_R + 1e-10)
                mse_recon_L = mse_recon_L[0].cpu().numpy()
                mse_recon_R = mse_recon_R[0].cpu().numpy()

                mse_sym_L = (sum_diff_sym_L / (sum_mask_L + 1e-10))[0].cpu().numpy()
                mse_sym_R = (sum_diff_sym_R / (sum_mask_R + 1e-10))[0].cpu().numpy()

                lambda_recon = getattr(self.opts, 'lambda_recon_cost', 2.0)

                total_left_cost = mse_L + mse_sym_L + lambda_recon * mse_recon_L
                total_right_cost = mse_R + mse_sym_R + lambda_recon * mse_recon_R

                # if too few valid pixels, we mark as invalid then later will be set to above max cost
                # without this there are a lot of false matches
                total_pixels = self.C * R * R
                validity_ratio_L = (sum_mask_L[0] / total_pixels).cpu().numpy()
                validity_ratio_R = (sum_mask_R[0] / total_pixels).cpu().numpy()
                
                min_valid_ratio = 0.5
                total_left_cost = np.where(validity_ratio_L < min_valid_ratio, -1.0, total_left_cost)
                total_right_cost = np.where(validity_ratio_R < min_valid_ratio, -1.0, total_right_cost)
               
                cost_volume_left[:, :, i] = total_left_cost
                cost_volume_right[:, :, i] = total_right_cost

        return cost_volume_left, cost_volume_right, disparity_candidates

    def apply_sgm(self, cost_volume, left=True):
        """
        Apply Semi-Global Matching to the cost volume.
        
        Args:
            cost_volume: numpy array of shape [H', W', D]
            
        Returns:
            disparity_indices: numpy array of shape [H', W'] with optimal disparity index per patch
        """
        from sgm import aggregate_costs, select_disparity, Parameters, Paths
        
        P1 = getattr(self.opts, 'sgm_P1', 20)
        P2 = getattr(self.opts, 'sgm_P2', 200)
        D = cost_volume.shape[2]
        
        # sgm params, NOTE P2 is still used but just scaled by boundary map
        parameters = Parameters(max_disparity=D, P1=P1, P2=P2)
        paths = Paths()
        
        # replace invalid with a high cost value
        invalid_mask = cost_volume < 0
        valid_costs = cost_volume[~invalid_mask]
        if len(valid_costs) > 0:
            max_valid_cost = valid_costs.max()
            cost_volume = np.where(invalid_mask, max_valid_cost * 1.5, cost_volume)
        
        # normalize cost volume and scale up
        cost_min = cost_volume.min()
        cost_max = cost_volume.max()
        cost_scaled = ((cost_volume - cost_min) / (cost_max - cost_min + 1e-8) * 1024).astype(np.uint32)
        
        # generate P2 map by scaling base P2 by boundary strength
        # higher boundary strength = lower P2 = easier to transition disparity
        alpha = getattr(self.opts, 'sgm_alpha_boundary', 2.0)
        if left:
            p2_map = np.maximum(P2 * np.exp(-alpha * self.left_boundary_strength), 50).astype(np.uint32)
        else:
            p2_map = np.maximum(P2 * np.exp(-alpha * self.right_boundary_strength), 50).astype(np.uint32)
        
        print('Running SGM aggregation...')
        aggregation_volume = aggregate_costs(cost_scaled, parameters, paths, p2_map=p2_map)
        disparity_indices = select_disparity(aggregation_volume)
        
        return disparity_indices

    def step_cost_volume(self, iteration, end_factor=1):
        """
        Same as orig step function, but using parallel initialization for both left and right images
        """

        # Linearly increase lambda from 0 to lambda_boundary_final and lambda_color_final
        if self.opts.num_refinement_iters <= 1:
            factor = 0.0
        else:
            factor = max([0, (iteration - self.opts.num_initialization_iters) / (self.opts.num_refinement_iters - 1)])
            factor = end_factor * factor
            
        lmbda_boundary = factor * self.opts.lambda_boundary_final
        lmbda_color    = factor * self.opts.lambda_color_final

        if iteration < self.opts.num_initialization_iters or \
               (iteration - self.opts.num_initialization_iters + 1) % self.opts.greedy_step_every_iters == 0:
            # Parallel initialization for both left and right images
            self.initialization_step_LEFT(lmbda_boundary, lmbda_color)
            self.initialization_step_RIGHT(lmbda_boundary, lmbda_color)
        else:
            # Parallel refinement for both left and right images
            self.refinement_step_LEFT(lmbda_boundary, lmbda_color)
            self.refinement_step_RIGHT(lmbda_boundary, lmbda_color)

    def upsample_disp(self, disparity_map):
        """
        Returns:
            [1, 1, H, W] upsampled disparity map
        """
        d_tensor = torch.as_tensor(disparity_map, device=dev, dtype=torch.float32).view(1, 1, self.H_patches, self.W_patches)
        d_patches = d_tensor.unsqueeze(-3).unsqueeze(-3).expand(-1, -1, self.opts.R, self.opts.R, -1, -1)
        return self.local2global(d_patches)
    
    def optimize_with_cost_volume(self):
        """
        Optimize FoJ representation, build cost volume, and apply SGM to recover disparity map
        """

        for iteration in tqdm(range(self.num_iters), desc='Pass 1: Optimizing FoJ'):
            self.step_cost_volume(iteration, end_factor=1)

        # build cost volume
        cost_volume_L, cost_volume_R, disparity_candidates = self.build_cost_volume()
        disparity_indices_L = self.apply_sgm(cost_volume_L, left=True)
        disparity_map_prev_pass_L = disparity_candidates[disparity_indices_L]  # [H', W'] in pixels
        
        # compute right view disparity for occlusion masking
        disparity_indices_R = self.apply_sgm(cost_volume_R, left=False)

        # pass 1 maps for debugging
        self.pass1_disparity_map = disparity_map_prev_pass_L.copy()
        self.pass1_global_boundaries = self.global_boundaries.detach().clone()
        self.pass1_global_right_boundaries = self.global_right_boundaries.detach().clone()
        self.pass1_global_image = self.global_image.detach().clone()
        self.pass1_global_right_image = self.global_right_image.detach().clone()
        
        disparity_normalized = disparity_map_prev_pass_L / (self.opts.R / 2.0)
        self.disparity.data = torch.tensor(disparity_normalized, device=dev, 
                                        dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # save right for later occlusion masking
        disparity_values_R = disparity_candidates[disparity_indices_R]
        disparity_normalized_R = disparity_values_R / (self.opts.R / 2.0)
        self.disparity_R.data = torch.tensor(disparity_normalized_R, device=dev, 
                                        dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
    def initialization_step_LEFT(self, lmbda_boundary=0, lmbda_color=0, disable_multi_pass=True):
        """
        Perform a single coordinate descent step (using Algorithm 2 from the paper).
        Implements a heuristic for searching along the three junction angles after updating each of
        the five parameters. The original value is included in the search, so the extra step is
        guaranteed to obtain a better (or equally-good) set of parameters.

        Inputs
        ------
        lmbda_boundary    Spatial consistency boundary loss weight
        lmbda_color       Spatial consistency color loss weight
        """
        params = torch.cat([self.angles, self.x0y0], dim=1).detach()

        # Run one step of Algorithm 2, sequentially improving each coordinate
        for i in range(5):
            # Repeat the set of parameters `nvals` times along 0th dimension
            params_query = params.repeat(self.opts.nvals, 1, 1, 1)
            param_range = self.angle_range if i < 3 else self.x0y0_range
            params_query[:, i, :, :] = params_query[:, i, :, :] + param_range.view(-1, 1, 1)
            best_ind = self.get_best_inds(params_query, lmbda_boundary, lmbda_color, disable_multi_pass=disable_multi_pass)

            # Update parameters
            params[0, i, :, :] = params_query[best_ind.view(1, self.H_patches, self.W_patches),
                                              i,
                                              torch.arange(self.H_patches).view(1, -1, 1),
                                              torch.arange(self.W_patches).view(1, 1, -1)]

        # Heuristic for accelerating convergence (not necessary but sometimes helps):
        # Update x0 and y0 along the three optimal angles (search over a line passing through current x0, y0)
        for i in range(3):
            params_query = params.repeat(self.opts.nvals, 1, 1, 1)
            params_query[:, 3, :, :] = params[:, 3, :, :] + torch.cos(params[:, i, :, :]) * self.x0y0_range.view(-1, 1, 1)
            params_query[:, 4, :, :] = params[:, 4, :, :] + torch.sin(params[:, i, :, :]) * self.x0y0_range.view(-1, 1, 1)
            best_ind = self.get_best_inds(params_query, lmbda_boundary, lmbda_color, disable_multi_pass=disable_multi_pass)

            # Update vertex positions of parameters
            for j in range(3, 5):
                params[:, j, :, :] = params_query[best_ind.view(1, self.H_patches, self.W_patches),
                                                  j,
                                                  torch.arange(self.H_patches).view(1, -1, 1),
                                                  torch.arange(self.W_patches).view(1, 1, -1)]

        # Update angles and vertex position using the best values found
        self.angles.data = params[:, :3, :, :].data
        self.x0y0.data   = params[:, 3:, :, :].data
        
        # Update global boundaries and image
        dists, colors, patches = self.get_dists_and_patches(params, lmbda_color)
        self.global_image      = self.local2global(patches)
        self.global_boundaries = self.local2global(self.dists2boundaries(dists))

    def refinement_step_LEFT(self, lmbda_boundary, lmbda_color):
        """
        Same as original refinement step but using only first two optimizers (left angles and x0y0)
        """
        params = torch.cat([self.angles, self.x0y0], dim=1)

        # Compute distance functions, colors, and junction patches
        dists, colors, patches = self.get_dists_and_patches(params, lmbda_color)
        
        # Compute loss
        loss = self.get_loss(dists, colors, patches, lmbda_boundary, lmbda_color).mean()

        # only optimize left img
        self.optimizers[0].zero_grad()
        self.optimizers[1].zero_grad()
        loss.backward()
        self.optimizers[0].step()
        self.optimizers[1].step()
        
        # Update global boundaries and image
        dists, colors, patches = self.get_dists_and_patches(params, lmbda_color)
        self.global_image      = self.local2global(patches)
        self.global_boundaries = self.local2global(self.dists2boundaries(dists))

    def get_best_inds_RIGHT(self, params, lmbda_boundary, lmbda_color, disable_multi_pass=True):
        """
        Same as get_best_inds but for right image
        """
        if self.opts.parallel_mode:
            dists, colors, smooth_patches = self.get_right_dists_and_patches(params, lmbda_color)
            loss_per_patch = self.get_right_loss(dists, colors, smooth_patches, lmbda_boundary, lmbda_color, disable_multi_pass=disable_multi_pass, params=params)
            best_ind = loss_per_patch.argmin(dim=0)

        else:
            # First initialize tensors
            best_ind            = torch.zeros(self.H_patches, self.W_patches, device=dev, dtype=torch.int64)
            best_loss_per_patch = torch.zeros(self.H_patches, self.W_patches, device=dev) + 1e10

            # Now fill tensors by iterating over the junction dimension and choosing the best junction parameters
            for n in range(params.shape[0]):
                dists, colors, smooth_patches = self.get_right_dists_and_patches(params[n:n+1, :, :, :], lmbda_color)

                loss_per_patch = self.get_right_loss(dists, colors, smooth_patches, lmbda_boundary, lmbda_color, disable_multi_pass=disable_multi_pass, params=params[n:n+1, :, :, :])

                improved_inds       = loss_per_patch[0] < best_loss_per_patch
                best_ind            = torch.where(improved_inds, torch.tensor(n, device=dev, dtype=torch.int64), best_ind)
                best_loss_per_patch = torch.where(improved_inds, loss_per_patch, best_loss_per_patch)

        return best_ind

    def initialization_step_RIGHT(self, lmbda_boundary=0, lmbda_color=0, disable_multi_pass=True):
        """
        Same as coordinate descent initialization_step but for right image
        """
        params = torch.cat([self.right_angles, self.right_x0y0], dim=1).detach()

        # Run one step of Algorithm 2, sequentially improving each coordinate
        for i in range(5):
            # Repeat the set of parameters `nvals` times along 0th dimension
            params_query = params.repeat(self.opts.nvals, 1, 1, 1)
            param_range = self.angle_range if i < 3 else self.x0y0_range
            params_query[:, i, :, :] = params_query[:, i, :, :] + param_range.view(-1, 1, 1)
            best_ind = self.get_best_inds_RIGHT(params_query, lmbda_boundary, lmbda_color, disable_multi_pass=disable_multi_pass)

            # Update parameters
            params[0, i, :, :] = params_query[best_ind.view(1, self.H_patches, self.W_patches),
                                              i,
                                              torch.arange(self.H_patches).view(1, -1, 1),
                                              torch.arange(self.W_patches).view(1, 1, -1)]

        # Heuristic for accelerating convergence (not necessary but sometimes helps):
        # Update x0 and y0 along the three optimal angles (search over a line passing through current x0, y0)
        for i in range(3):
            params_query = params.repeat(self.opts.nvals, 1, 1, 1)
            params_query[:, 3, :, :] = params[:, 3, :, :] + torch.cos(params[:, i, :, :]) * self.x0y0_range.view(-1, 1, 1)
            params_query[:, 4, :, :] = params[:, 4, :, :] + torch.sin(params[:, i, :, :]) * self.x0y0_range.view(-1, 1, 1)
            best_ind = self.get_best_inds_RIGHT(params_query, lmbda_boundary, lmbda_color, disable_multi_pass=disable_multi_pass)

            # Update vertex positions of parameters
            for j in range(3, 5):
                params[:, j, :, :] = params_query[best_ind.view(1, self.H_patches, self.W_patches),
                                                  j,
                                                  torch.arange(self.H_patches).view(1, -1, 1),
                                                  torch.arange(self.W_patches).view(1, 1, -1)]

        # Update angles and vertex position using the best values found
        self.right_angles.data = params[:, :3, :, :].data
        self.right_x0y0.data   = params[:, 3:, :, :].data
        
        # Update global boundaries and image
        dists, colors, patches = self.get_right_dists_and_patches(params, lmbda_color)
        self.global_right_image      = self.local2global(patches)
        self.global_right_boundaries = self.local2global(self.dists2boundaries(dists))

    def refinement_step_RIGHT(self, lmbda_boundary, lmbda_color):
        """
        Refinement step for RIGHT image using optimizers 2 and 3 (right angles and x0y0)
        """
        params = torch.cat([self.right_angles, self.right_x0y0], dim=1)

        dists, colors, patches = self.get_right_dists_and_patches(params, lmbda_color)
        
        loss = self.get_right_loss(dists, colors, patches, lmbda_boundary, lmbda_color).mean()

        # only optimize right img
        self.optimizers[2].zero_grad()
        self.optimizers[3].zero_grad()
        loss.backward()
        self.optimizers[2].step()
        self.optimizers[3].step()
        
        # Update global boundaries and image
        dists, colors, patches = self.get_right_dists_and_patches(params, lmbda_color)
        self.global_right_image      = self.local2global(patches)
        self.global_right_boundaries = self.local2global(self.dists2boundaries(dists))