import os
import cv2
import torch
import numpy as np
from PIL import Image

import torchvision.transforms.functional as TF

# Full kernels
FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), np.uint8)

# 7x7 diamond kernel
DIAMOND_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

# 9x9 diamond kernel
DIAMOND_KERNEL_9 = np.asarray(
    [
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
    ], dtype=np.uint8)

# 13x13 diamond kernel
DIAMOND_KERNEL_13 = np.asarray(
    [
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    ], dtype=np.uint8)

def read_depth(path):
    if path.endswith('.png'):
        depth = cv2.imread(path, -1)
        if depth is None:
            print(f"Warning: Could not read PNG image at {path}. Returning None.")
            return None
        depth = depth.astype(np.float32) / 256.0
        return depth
    elif path.endswith('.bin'):
        try:
            # For KITTI velodyne .bin files, they contain 4 floats: x, y, z, intensity
            obj = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
            return obj # Return the raw velodyne points
        except Exception as e:
            print(f"Error reading BIN file at {path}: {e}. Returning None.")
            return None
    else:
        print(f"Unsupported file extension for depth reading: {path}. Returning None.")
        return None

def read_rgb(path):
    img = Image.open(path)
    return img

def read_calib_file(filepath):
    """
    Read KITTI calibration file and return dictionary of calibration matrices
    """
    calib_data = {}
    
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
                
            # Split by colon to separate key and values
            if ':' in line:
                key, values_str = line.split(':', 1)
                key = key.strip()
                values_str = values_str.strip()
                
                # Split values by whitespace and convert to float
                try:
                    values = [float(x) for x in values_str.split()]
                    calib_data[key] = np.array(values, dtype=np.float32)
                except ValueError as e:
                    print(f"Warning: Could not parse line '{line}': {e}")
                    continue
    
    return calib_data

def Crop(img, h_init, w_init, rheight, rwidth):

    # crop the image
    img = img[h_init:h_init+rheight, w_init:w_init+rwidth]

    return img

def count_parameters(model):
    """
    Counts the number of trainable parameters in a PyTorch model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def remove_moudle(state_dict):
    """
    Remove 'module.' prefix from state_dict keys for DataParallel models.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def count_validpoint(output, sample, t_valid=0.0001):
    with torch.no_grad():
        pred, gt = output.detach(), sample.detach()
        mask = gt > t_valid
        num_valid = mask.sum()
    return num_valid

def get_random_hw(input_height, input_width, real_height, real_width):
    # Randomly crop
    h_init = np.random.randint(0, real_height - input_height)
    w_init = np.random.randint(0, real_width - input_width)
    rheight, rwidth = input_height, input_width
    return real_height, real_width, h_init, w_init

def color_jitter(img, brightness, contrast, saturation, hue):
    """
    Apply color jitter to an image.
    This function is adapted from the provided utility, but the original implementation
    with PCA jitter is retained as it's more specific to the codebase.
    """
    # METHOD1: From torchvision.transforms.ColorJitter
    # img = TF.adjust_brightness(img, brightness)
    # img = TF.adjust_contrast(img, contrast)
    # img = TF.adjust_saturation(img, saturation)

    # METHOD2
    # borrow from https://github.com/kujason/avod/blob/master/avod/datasets/kitti/kitti_aug.py
    img_np = np.array(img)
    pca = compute_pca(img_np)
    img_np = add_pca_jitter(img_np, pca)
    img = Image.fromarray(img_np, 'RGB')

    return img

def compute_pca(image):
    """
    calculate PCA of image
    """

    reshaped_data = image.reshape(-1, 3)
    reshaped_data = (reshaped_data / 255.0).astype(np.float32)
    covariance = np.cov(reshaped_data.T)
    e_vals, e_vecs = np.linalg.eigh(covariance)
    pca = np.sqrt(e_vals) * e_vecs
    return pca


def outlier_removal(lidar):
    # sparse_lidar = np.squeeze(lidar)
    threshold = 1.0
    sparse_lidar = lidar
    valid_pixels = (sparse_lidar > 0.1).astype(np.float)
    print(sparse_lidar.shape)
    # print(len(sparse_lidar))
    lidar_sum_7 = cv2.filter2D(sparse_lidar, -1, DIAMOND_KERNEL_7)
    lidar_count_7 = cv2.filter2D(valid_pixels, -1, DIAMOND_KERNEL_7)

    lidar_aveg_7 = lidar_sum_7 / (lidar_count_7 + 0.00001)
    potential_outliers_7 = ((sparse_lidar - lidar_aveg_7) > threshold).astype(np.float)

    lidar_sum_9 = cv2.filter2D(sparse_lidar, -1, DIAMOND_KERNEL_9)
    lidar_count_9 = cv2.filter2D(valid_pixels, -1, DIAMOND_KERNEL_9)

    lidar_aveg_9 = lidar_sum_9 / (lidar_count_9 + 0.00001)
    potential_outliers_9 = ((sparse_lidar - lidar_aveg_9) > threshold).astype(np.float)

    lidar_sum_13 = cv2.filter2D(sparse_lidar, -1, DIAMOND_KERNEL_13)
    lidar_count_13 = cv2.filter2D(valid_pixels, -1, DIAMOND_KERNEL_13)

    lidar_aveg_13 = lidar_sum_13 / (lidar_count_13 + 0.00001)
    potential_outliers_13 = ((sparse_lidar - lidar_aveg_13) > threshold).astype(np.float)

    potential_outliers = potential_outliers_7 + potential_outliers_9 + potential_outliers_13
    lidar_cleared = (sparse_lidar * (1 - potential_outliers)).astype(np.float32)

    return lidar_cleared, potential_outliers


def project_velodyne_to_depth_map(velo_points, P_rect_xx, Tr_velo_to_cam, R0_rect, img_height, img_width, max_depth=80.0):
    """
    Projects raw velodyne points to a sparse depth map.

    Args:
        velo_points (np.ndarray): Raw velodyne points (N, 4) [x, y, z, intensity].
        P_rect_xx (np.ndarray): Camera projection matrix (3, 4) for rectified image (e.g., P2 or P3).
        Tr_velo_to_cam (np.ndarray): Velodyne to camera transformation matrix (3, 4).
        R0_rect (np.ndarray): Rectification matrix (3, 3) for camera 00 (reference camera).
        img_height (int): Target image height.
        img_width (int): Target image width.
        max_depth (float): Maximum depth value to consider.

    Returns:
        np.ndarray: Sparse depth map (1, H, W) in meters.
    """
    # Handle cases where velo_points is None or empty
    if velo_points is None or velo_points.shape[0] == 0:
        print("Warning: velo_points is None or empty. Returning a zero-filled depth map.")
        return np.zeros((1, img_height, img_width), dtype=np.float32)

    # 1. Transform points from velodyne to camera frame (cam00, unrectified)
    points_cam00_unrect = np.dot(Tr_velo_to_cam[:3, :3], velo_points[:, :3].T).T + Tr_velo_to_cam[:3, 3] # (N, 3)

    # 2. Project points from camera 00 unrectified to rectified camera image plane (e.g., cam02 or cam03)
    points_cam_homogeneous = np.hstack((points_cam00_unrect, np.ones((points_cam00_unrect.shape[0], 1)))) # (N, 4)

    # Project to 2D image plane using P_rect_xx
    img_points_raw = np.dot(P_rect_xx, points_cam_homogeneous.T).T # (N, 3) -> (u_raw, v_raw, Z_cam)
    
    # Normalize by the third coordinate (Z_cam) to get pixel coordinates
    # Filter out points that have non-positive depth (behind the camera or at sensor plane)
    valid_points_mask = img_points_raw[:, 2] > 0.001 
    
    img_points_norm = img_points_raw[valid_points_mask]
    
    # If no valid points remain after depth filtering, return a zero map
    if img_points_norm.shape[0] == 0:
        print("Warning: No valid points after depth filtering. Returning a zero-filled depth map.")
        return np.zeros((1, img_height, img_width), dtype=np.float32)

    u = (img_points_norm[:, 0] / img_points_norm[:, 2]).astype(int)
    v = (img_points_norm[:, 1] / img_points_norm[:, 2]).astype(int)
    depth_values = img_points_norm[:, 2] # This is Z_cam, which is the depth.

    # Filter points within image bounds and max depth
    valid_pixels_mask = (u >= 0) & (u < img_width) & \
                        (v >= 0) & (v < img_height) & \
                        (depth_values > 0) & (depth_values <= max_depth)

    u_final = u[valid_pixels_mask]
    v_final = v[valid_pixels_mask]
    depth_final = depth_values[valid_pixels_mask]

    # If no valid pixels remain after all filtering, return a zero map
    if len(u_final) == 0:
        print("Warning: No valid pixels after all filtering. Returning a zero-filled depth map.")
        return np.zeros((1, img_height, img_width), dtype=np.float32)

    # Create sparse depth map
    temp_depth_map = np.full((img_height, img_width), np.inf, dtype=np.float32)

    # Populate the depth map. If multiple points project to the same pixel, take the closest one.
    for i in range(len(u_final)):
        current_depth = depth_final[i]
        if current_depth < temp_depth_map[v_final[i], u_final[i]]:
            temp_depth_map[v_final[i], u_final[i]] = current_depth
    
    sparse_depth_map = np.where(temp_depth_map == np.inf, 0.0, temp_depth_map)

    return np.expand_dims(sparse_depth_map, 0) # Add channel dimension (1, H, W)

def add_pca_jitter(img_data, pca):
    """
    add a multiple of principal components with Gaussian noise
    """
    new_img_data = np.copy(img_data).astype(np.float32) / 255.0
    magnitude = np.random.randn(3) * 0.1
    noise = (pca * magnitude).sum(axis=1)

    new_img_data = new_img_data + noise
    new_img_data = np.clip(new_img_data, 0.0, 1.0)
    new_img_data = (new_img_data * 255.0).astype(np.uint8)
    return new_img_data