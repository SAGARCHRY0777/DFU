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


def read_depth(filepath):
    """
    Reads depth data from a PNG or BIN file.
    If PNG, reads a 16-bit depth image and scales it.
    If BIN, reads raw velodyne points (N, 4) [x, y, z, intensity].
    """
    if filepath.endswith('.png'):
        depth_png = np.array(Image.open(filepath), dtype=int)
        # make sure we have a proper 16bit depth map (not 8bit)
        assert(np.max(depth_png) > 255)
        depth = depth_png.astype(np.float32) / 256. # Scale to meters
        return np.expand_dims(depth, 0) # Add channel dimension (1, H, W)
    elif filepath.endswith('.bin'):
        # Read raw velodyne point cloud data (x, y, z, intensity)
        # Returns (N, 4) numpy array
        velo_points = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)
        return velo_points
    else:
        raise ValueError(f"Unsupported depth file format: {filepath}")

def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line: # Skip empty lines
                continue
            if ':' not in line: # Skip lines without colon (e.g., comments)
                continue
            
            key, value_str = line.split(':', 1)
            key = key.strip() # Strip whitespace from key
            
            try:
                # Filter out empty strings from split, then convert to float
                # This handles cases like "1.0  2.0" -> ['1.0', '', '2.0']
                numeric_values = [float(x) for x in value_str.strip().split(' ') if x.strip()]
                
                # Only add to data if there are valid numeric values
                if numeric_values:
                    data[key] = np.array(numeric_values)
                else:
                    print(f"Warning: No valid numeric values found for key '{key}' in {filepath}. Line: '{line}'")
            except ValueError:
                print(f"Warning: Could not parse numeric values for key '{key}' in {filepath}. Line: '{line}'")
                # If conversion fails, this key will simply not be added to the dictionary.
                pass 
    return data

def outlier_removal(lidar):
    # sparse_lidar = np.squeeze(lidar)
    threshold = 1.0
    sparse_lidar = lidar
    valid_pixels = (sparse_lidar > 0.1).astype(np.float)

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
    # 1. Transform points from velodyne to camera frame (cam00, unrectified)
    # velo_points[:, :3] are (N, 3) xyz coordinates
    # Tr_velo_to_cam[:3, :3] is 3x3 rotation, Tr_velo_to_cam[:3, 3] is 3x1 translation
    points_cam00_unrect = np.dot(Tr_velo_to_cam[:3, :3], velo_points[:, :3].T).T + Tr_velo_to_cam[:3, 3] # (N, 3)

    # 2. Project points from camera 00 unrectified to rectified camera image plane (e.g., cam02 or cam03)
    # P_rect_xx is 3x4 and already incorporates the intrinsic matrix K and rectification for the specific camera.
    # We need homogeneous coordinates for points_cam00_unrect.
    points_cam_homogeneous = np.hstack((points_cam00_unrect, np.ones((points_cam00_unrect.shape[0], 1)))) # (N, 4)

    # Project to 2D image plane using P_rect_xx
    img_points_raw = np.dot(P_rect_xx, points_cam_homogeneous.T).T # (N, 3) -> (u_raw, v_raw, Z_cam)
    
    # Normalize by the third coordinate (Z_cam) to get pixel coordinates
    # Filter out points that have non-positive depth (behind the camera or at sensor plane)
    valid_points_mask = img_points_raw[:, 2] > 0.001 
    
    img_points_norm = img_points_raw[valid_points_mask]
    
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

    # Create sparse depth map
    # Initialize with a very large value to ensure the minimum depth is kept.
    temp_depth_map = np.full((img_height, img_width), np.inf, dtype=np.float32)

    # Populate the depth map. If multiple points project to the same pixel, take the closest one.
    for i in range(len(u_final)):
        current_depth = depth_final[i]
        # Only update if the current depth is smaller than the existing one (or if it's the first assignment)
        if current_depth < temp_depth_map[v_final[i], u_final[i]]:
            temp_depth_map[v_final[i], u_final[i]] = current_depth
    
    # Replace np.inf with 0 (or a designated 'no-data' value) for pixels with no points
    sparse_depth_map = np.where(temp_depth_map == np.inf, 0.0, temp_depth_map)

    return np.expand_dims(sparse_depth_map, 0) # Add channel dimension (1, H, W)

def get_sparse_depth(dep, num_spot):
    channel, height, width = dep.shape
    assert channel == 1
    idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)
    num_idx = len(idx_nnz)
    idx_sample = torch.randperm(num_idx)[:num_spot]
    idx_nnz = idx_nnz[idx_sample[:]]
    mask = torch.zeros((channel * height * width))
    mask[idx_nnz] = 1.0
    mask = mask.view((channel, height, width))
    dep_sp = dep * mask.type_as(dep)
    return dep_sp

def get_sparse_depth_grid(dep):
    """
    Simulate pincushion distortion:
    --stride:
    It controls the distance between neighbor spots7
    Suggest stride value:       5~10

    --dist_coef:
    It controls the curvature of the spot pattern
    Larger dist_coef distorts the pattern more.
    Suggest dist_coef value:    0 ~ 5e-5

    --noise:
    standard deviation of the spot shift
    Suggest noise value:        0 ~ 0.5
    """

    # Generate Grid points
    channel, img_h, img_w = dep.shape
    assert channel == 1

    stride = np.random.randint(5, 7)

    dist_coef = np.random.rand() * 4e-5 + 1e-5
    noise = np.random.rand() * 0.3

    x_odd, y_odd = np.meshgrid(np.arange(stride // 2, img_h, stride * 2), np.arange(stride // 2, img_w, stride))
    x_even, y_even = np.meshgrid(np.arange(stride // 2 + stride, img_h, stride * 2), np.arange(stride, img_w, stride))
    x_u = np.concatenate((x_odd.ravel(), x_even.ravel()))
    y_u = np.concatenate((y_odd.ravel(), y_even.ravel()))
    x_c = img_h // 2 + np.random.rand() * 50 - 25
    y_c = img_w // 2 + np.random.rand() * 50 - 25
    x_u = x_u - x_c
    y_u = y_u - y_c

    # Distortion
    r_u = np.sqrt(x_u ** 2 + y_u ** 2)
    r_d = r_u + dist_coef * r_u ** 3
    num_d = r_d.size
    sin_theta = x_u / r_u
    cos_theta = y_u / r_u
    x_d = np.round(r_d * sin_theta + x_c + np.random.normal(0, noise, num_d))
    y_d = np.round(r_d * cos_theta + y_c + np.random.normal(0, noise, num_d))
    idx_mask = (x_d < img_h) & (x_d > 0) & (y_d < img_w) & (y_d > 0)
    x_d = x_d[idx_mask].astype('int')
    y_d = y_d[idx_mask].astype('int')

    spot_mask = np.zeros((img_h, img_w))
    spot_mask[x_d, y_d] = 1

    dep_sp = torch.zeros_like(dep)
    dep_sp[:, x_d, y_d] = dep[:, x_d, y_d]

    return dep_sp


def cut_mask(dep):
    _, h, w = dep.size()
    c_x = np.random.randint(h / 4, h / 4 * 3)
    c_y = np.random.randint(w / 4, w / 4 * 3)
    r_x = np.random.randint(h / 4, h / 4 * 3)
    r_y = np.random.randint(h / 4, h / 4 * 3)

    mask = torch.zeros_like(dep)
    min_x = max(c_x - r_x, 0)
    max_x = min(c_x + r_x, h)
    min_y = max(c_y - r_y, 0)
    max_y = min(c_y + r_y, w)
    mask[0, min_x:max_x, min_y:max_y] = 1

    return dep * mask


def get_sparse_depth_prop(dep, prop):
    channel, height, width = dep.shape

    assert channel == 1

    idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)

    num_idx = len(idx_nnz)
    num_sample = int(num_idx * prop)
    idx_sample = torch.randperm(num_idx)[:num_sample]

    idx_nnz = idx_nnz[idx_sample[:]]

    mask = torch.zeros((channel * height * width))
    mask[idx_nnz] = 1.0
    mask = mask.view((channel, height, width))

    dep_sp = dep * mask.type_as(dep)

    return dep_sp


def get_sparse_depthv2(dep, num_sample):
    channel, height, width = dep.shape
    assert channel == 1
    idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)
    num_idx = len(idx_nnz)
    idx_sample = torch.randperm(num_idx)[:num_sample]
    idx_nnz = idx_nnz[idx_sample[:]]
    mask = torch.zeros((channel * height * width))
    mask[idx_nnz] = 1.0
    mask = mask.view((channel, height, width))
    dep_sp = dep * mask.type_as(dep)
    return dep_sp

def read_rgb(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    return img_file

def Crop(img, h_init, w_init, crop_h, crop_w):
    return TF.crop(img, h_init, w_init, crop_h, crop_w)

def Hflip(img, flip):
    if flip > 0.5:
        return TF.hflip(img)
    else:
        return img

def ColorJitter(img):
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

def add_pca_jitter(img_data, pca):
    """
    add a multiple of principal components with Gaussian noise
    """
    new_img_data = np.copy(img_data).astype(np.float32) / 255.0
    magnitude = np.random.randn(3) * 0.1
    noise = (pca * magnitude).sum(axis=1)

    new_img_data = new_img_data + noise
    np.clip(new_img_data, 0.0, 1.0, out=new_img_data)
    new_img_data = (new_img_data * 255).astype(np.uint8)

    return new_img_data

def Rotation(img, degree):
    return TF.rotate(img, angle=degree)

def Resize(img, size, mode=None):
    if mode:
        return TF.resize(img, size, mode)
    else:
        return TF.resize(img, size)