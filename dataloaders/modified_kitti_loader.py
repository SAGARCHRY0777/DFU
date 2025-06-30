import torch.utils.data as data
import numpy as np
import os
import torch
from dataloaders.modified_path_and_transform import get_kittipaths, get_rgb, kittitransforms
from dataloaders.modified_NNfill import fill_in_fast
from dataloaders.modified_utils import read_depth, read_calib_file, outlier_removal, project_velodyne_to_depth_map
import cv2 # Import cv2 for image operations, especially if read_rgb uses it or for potential future use

class KittiDepth(data.Dataset):
    """A data loader for the Kitti dataset"""

    def __init__(self, split, args):
        self.args = args
        self.split = split
        self.paths = get_kittipaths(split, args)
        self.transforms = kittitransforms
        self.ipfill = fill_in_fast

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.paths['dep'])

    def __getraw__(self, index):
        dep_path = self.paths['dep'][index]
        rgb_path = self.paths['rgb'][index]
        calib_file_path = self.paths['K'][index]

        # Get RGB image
        rgb = get_rgb(index, self.paths, self.split, self.args) if \
            (rgb_path is not None) else None

        # Determine image dimensions from RGB for projection if available
        # Otherwise, fallback to args.val_h, args.val_w
        img_h, img_w = self.args.val_h, self.args.val_w
        if rgb is not None:
            if isinstance(rgb, np.ndarray):
                img_h, img_w = rgb.shape[0], rgb.shape[1]
            elif hasattr(rgb, 'size'): # PIL Image
                img_w, img_h = rgb.size # PIL Image.size is (width, height)


        # Read Ground Truth depth (if available)
        gt = read_depth(self.paths['gt'][index]) if \
            self.paths['gt'][index] is not None else None

        # Initialize calibration matrices and intrinsic parameters
        K_intrinsics_list = None
        P_rect_xx = None
        Tr_velo_to_cam = None
        R0_rect = None

        if calib_file_path is not None and os.path.exists(calib_file_path):
            calib_data = read_calib_file(calib_file_path)

            if 'image_2' in rgb_path or 'image_02' in rgb_path:
                if 'P2' in calib_data:
                    try:
                        p2_data = np.array(calib_data['P2'])
                        if p2_data.size == 12:
                            P_rect_xx = p2_data.reshape(3, 4)
                        else:
                            P_rect_xx = None
                    except Exception as e:
                        P_rect_xx = None
            elif 'image_3' in rgb_path or 'image_03' in rgb_path:
                if 'P3' in calib_data:
                    try:
                        p3_data = np.array(calib_data['P3'])
                        if p3_data.size == 12:
                            P_rect_xx = p3_data.reshape(3, 4)
                        else:
                            P_rect_xx = None
                    except Exception as e:
                        P_rect_xx = None

            # Populate K_intrinsics_list if P_rect_xx is found
            if P_rect_xx is not None:
                K_intrinsics_list = [P_rect_xx[0, 0], P_rect_xx[1, 1], P_rect_xx[0, 2], P_rect_xx[1, 2]]
            else:
                pass

            # Extract Tr_velo_to_cam
            if 'Tr_velo_to_cam' in calib_data:
                try:
                    tr_data = np.array(calib_data['Tr_velo_to_cam'])
                    if tr_data.size == 12:
                        Tr_velo_to_cam = tr_data.reshape(3, 4)
                    else:
                        Tr_velo_to_cam = None
                except Exception as e:
                    Tr_velo_to_cam = None
            else:
                pass

            # Extract R0_rect
            if 'R0_rect' in calib_data:
                try:
                    r0_data = np.array(calib_data['R0_rect'])
                    if r0_data.size == 9:
                        R0_rect = r0_data.reshape(3, 3)
                    else:
                        R0_rect = np.eye(3)
                except Exception as e:
                    R0_rect = np.eye(3)
            else:
                R0_rect = np.eye(3)
        else:
            pass

        # Process depth/velodyne data
        dep = None
        if dep_path is not None:
            if dep_path.endswith('.bin') and P_rect_xx is not None and Tr_velo_to_cam is not None and R0_rect is not None:
                # Read raw velodyne points
                velo_points = read_depth(dep_path)

                # Project velodyne points to sparse depth map
                dep = project_velodyne_to_depth_map(
                    velo_points, P_rect_xx, Tr_velo_to_cam, R0_rect,
                    img_h, img_w, max_depth=self.args.max_depth
                )
                print(f"Projected depth shape: {dep.shape}")  # Debugging line
                if dep is not None:
                    print(f"Depth shape after projection before squeeze: {dep.shape}")  # Debugging line
                    # Ensure dep is 2D numpy array (H, W) for transformations
                    dep = dep.squeeze() # Remove channel dim if present from projection
                    print(f"Depth shape after squeeze:==",dep.shape)  # Debugging line
                    if dep.ndim == 0: # Handle case where squeeze makes it scalar
                        print("the shape of the depth is 0D, reshaping to 1x1")
                        dep = np.array([[dep]]) # Make it a 1x1 array
                        print(f"Depth shape after reshaping: {dep.shape}")  # Debugging line
                    elif dep.ndim == 1: # Handle 1D array, make it 2D
                        print("the shape of the depth is 1D, reshaping to 1xW")
                        dep = np.expand_dims(dep, 0)
                        print(f"Depth shape after reshaping: {dep.shape}")  # Debugging line

            elif dep_path.endswith('.png'):
                # Read PNG depth images directly
                dep = read_depth(dep_path)
                print(f"Depth shape after reading PNG: {dep.shape}")  # Debugging line
            else:
                dep = None
        else:
            pass

        return dep, gt, K_intrinsics_list, rgb, dep_path

    def __getitem__(self, index):
        dep, gt, K, rgb, paths = self.__getraw__(index)

        # print(f"Initial dep type in __getitem__: {type(dep)}")
        # if isinstance(dep, np.ndarray):
        #     print(f"Initial dep shape in __getitem__: {dep.shape}")

        # Ensure GT is treated similarly to dep for transformations
        if gt is not None and not isinstance(gt, np.ndarray):
            # If gt is not numpy, convert it (e.g., if it's a PIL image)
            gt = np.array(gt)
            if gt.ndim == 3 and gt.shape[2] == 1:
                gt = gt.squeeze(2) # Remove channel dim if it's (H,W,1)

        # Apply transformations
        dep_transformed, gt_transformed, K_transformed, rgb_transformed = \
            self.transforms(self.split, self.args, dep, gt, K, rgb)
        print("the shape of the depth after transformation:", dep_transformed.shape)  # Debugging line

        # Check if transformed depth is empty or invalid
        if dep_transformed is None or dep_transformed.numel() == 0:
            print("Warning: Depth tensor is None or empty after transformation.")
            print(f"Depth tensor shape: {dep_transformed.shape if dep_transformed is not None else 'None'}")
            # Fallback to zero tensors if data is invalid after transformation
            dep_clear_torch = torch.zeros(1, self.args.val_h, self.args.val_w, dtype=torch.float32)
            print(f"Depth tensor after fallback: {dep_clear_torch.shape}")  # Debugging line
            dep_ip_torch = torch.zeros(self.args.val_h, self.args.val_w, dtype=torch.float32)
            print(f"IP tensor after fallback: {dep_ip_torch.shape}")  # Debugging line
            # K might also be invalid if source image was too small, reset it.
            K_transformed_tensor = torch.zeros(4, dtype=torch.float32) # Convert K to a tensor here
        else:
            # Ensure dep_transformed is a 2D numpy array for outlier_removal and ipfill
            if torch.is_tensor(dep_transformed):
                print("shape of dep_transformed before numpy conversion:", dep_transformed.shape)  # Debugging line
                dep_np = dep_transformed.squeeze(0).cpu().numpy() # Remove batch/channel dim if present
                print(f"Converted dep_transformed to numpy with shape: {dep_np.shape}")  # Debugging line
            else: # If it's already numpy, just ensure it's 2D
                dep_np = dep_transformed
                print(f"dep_transformed is already numpy with shape: {dep_np.shape}")  # Debugging line
                if dep_np.ndim == 3 and dep_np.shape[0] == 1:
                    print("the shape of the depth is 3D with 1 channel, squeezing to 2D")
                    print(f"Depth shape before squeeze: {dep_np.shape}")  # Debugging line
                    dep_np = dep_np.squeeze(0)
                    print(f"Depth shape after squeeze: {dep_np.shape}")  # Debugging line
                elif dep_np.ndim == 3 and dep_np.shape[2] == 1: # (H, W, 1) case
                    print("the shape of the depth is 3D with 1 channel, squeezing to 2D")
                    print(f"Depth shape before squeeze: {dep_np.shape}")  # Debugging line
                    dep_np = dep_np.squeeze(2)
                    print(f"Depth shape after squeeze: {dep_np.shape}")

            # Check again if dep_np is empty after all processing
            if dep_np.size == 0 or dep_np.ndim != 2:
                # This should ideally not happen with correct cropping logic, but as a safeguard
                print(f"Warning: Depth numpy array is empty or not 2D after transformations. Shape: {dep_np.shape}")
                dep_clear_torch = torch.zeros(1, self.args.val_h, self.args.val_w, dtype=torch.float32)
                print(f"Depth tensor after fallback: {dep_clear_torch.shape}")  # Debugging line
                dep_ip_torch = torch.zeros(self.args.val_h, self.args.val_w, dtype=torch.float32)
                print(f"IP tensor after fallback: {dep_ip_torch.shape}")
            else:
                print(f"Processed dep_np shape: {dep_np.shape}")  # Debugging line
                print("this is befor the outlier removal")
                print("shape of dep_np before outlier removal:", dep_np.shape)  # Debugging line
                dep_clear, _ = outlier_removal(dep_np)
                print("this is after the outlier removal")
                print("shape of dep_clear after outlier removal:", dep_clear.shape)  # Debugging line
                dep_clear_torch = torch.from_numpy(np.expand_dims(dep_clear, 0)).float() # Add channel back
                print(f"Converted dep_clear to torch with shape: {dep_clear_torch.shape}")  # Debugging line

                dep_np_ip = np.copy(dep_np)
                dep_ip = self.ipfill(dep_np_ip, max_depth=self.args.max_depth,
                                    extrapolate=True, blur_type='gaussian')
                dep_ip_torch = torch.from_numpy(dep_ip).float()

            # Ensure K_transformed is a list of numbers before converting to tensor
            if isinstance(K_transformed, list):
                K_transformed_tensor = torch.tensor(K_transformed, dtype=torch.float32)
            else: # Fallback if K_transformed is not a list (e.g., None or invalid)
                K_transformed_tensor = torch.zeros(4, dtype=torch.float32)


        # Ensure rgb_transformed has 3 channels and is a tensor
        if rgb_transformed is None or rgb_transformed.numel() == 0:
            rgb_transformed = torch.zeros(3, self.args.val_h, self.args.val_w, dtype=torch.float32)
        elif torch.is_tensor(rgb_transformed) and rgb_transformed.ndim == 2:
            rgb_transformed = rgb_transformed.unsqueeze(0).repeat(3, 1, 1) # If grayscale, convert to 3 channels
        elif torch.is_tensor(rgb_transformed) and rgb_transformed.ndim == 3 and rgb_transformed.shape[0] == 1:
             rgb_transformed = rgb_transformed.repeat(3, 1, 1) # If single channel, repeat to 3

        # Ensure gt_transformed is a tensor and has a channel dimension
        if gt_transformed is None or gt_transformed.numel() == 0:
            gt_transformed_tensor = torch.zeros(1, self.args.val_h, self.args.val_w, dtype=torch.float32)
        elif torch.is_tensor(gt_transformed):
            if gt_transformed.ndim == 2:
                gt_transformed_tensor = gt_transformed.unsqueeze(0)
            else:
                gt_transformed_tensor = gt_transformed
        else: # Assuming it's numpy after transforms if not None
            gt_transformed_tensor = torch.from_numpy(np.expand_dims(gt_transformed, 0)).float()

        candidates = {
            'dep': dep_transformed,
            'dep_clear': dep_clear_torch,
            'dep_ip': dep_ip_torch,
            'rgb': rgb_transformed,
            'K': K_transformed_tensor, # Use the converted tensor
            'gt': gt_transformed_tensor,
            'paths': paths # 'paths' should ideally not be moved to CPU/GPU if it's a string.
                           # If it's only used for logging, it can remain a string.
                           # If it's processed later and expected to be a tensor, convert it.
                           # For now, it's safer to keep it out of the .cpu() call in test.py
        }

        return candidates