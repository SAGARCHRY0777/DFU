import cv2
import numpy as np
from scipy.ndimage.measurements import label

# Full kernels
FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
FULL_KERNEL_13 = np.ones((13, 13), np.uint8)
FULL_KERNEL_25 = np.ones((25, 25), np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), np.uint8)

# 3x3 cross kernel
CROSS_KERNEL_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)

# 5x5 cross kernel
CROSS_KERNEL_5 = np.asarray(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 5x5 diamond kernel
DIAMOND_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 7x7 cross kernel
CROSS_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

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

def get_filled_lidar(sparse_lidar, max_depth=100.0,
                     extrapolate=True, blur_type='gaussian'):

    if extrapolate:
        sparse_lidar = extrapolate_valid_pixels(sparse_lidar)

    # Median blur
    depth_map = sparse_lidar.astype('float32') # Cast a float64 image to float32
    depth_map = cv2.medianBlur(depth_map, 5)
    depth_map = depth_map.astype('float64') # Cast a float32 image to float64
    #
    # Bilateral or Gaussian blur
    if blur_type == 'bilateral':
        # Bilateral blur
        depth_map = depth_map.astype('float32')
        depth_map = cv2.bilateralFilter(depth_map, 5, 1.5, 2.0)
        depth_map = depth_map.astype('float64')
    elif blur_type == 'gaussian':
        # Gaussian blur
        valid_pixels = (depth_map > 0.1)
        blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
        depth_map[valid_pixels] = blurred[valid_pixels]

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    # fill zero value
    mask = (depth_map <= 0.1)
    if np.sum(mask) != 0:
        labeled_array, num_features = label(mask)
        for i in range(num_features):
            index = i + 1
            m = (labeled_array == index)
            m_dilate1 = cv2.dilate(1.0*m, FULL_KERNEL_3, iterations=1)
            m_dilate2 = cv2.dilate(1.0*m_dilate1, FULL_KERNEL_5, iterations=1)
            m_dilate3 = cv2.dilate(1.0*m_dilate2, FULL_KERNEL_7, iterations=1)
            m_dilate4 = cv2.dilate(1.0*m_dilate3, FULL_KERNEL_9, iterations=1)

            depth_map_copy = np.copy(depth_map)
            depth_map_copy[m] = 0.0

            depth_map[m] = depth_map_copy[m_dilate4 > 0].mean()

    # Invert back
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    return depth_map

def extrapolate_valid_pixels(depth,
                             dilations=(1, 2, 4, 8, 16),
                             kernels=(DIAMOND_KERNEL_5,
                                      DIAMOND_KERNEL_7, DIAMOND_KERNEL_9,
                                      DIAMOND_KERNEL_13),
                             max_depth=100.0):
    """
    Extrapolates sparse depth map to make it dense using dilated convolutions.
    The holes in the input depth map are filled with values from valid nearby pixels.
    It is done by repeatedly applying a dilated convolution with increasing dilation rates.
    """
    # Create a mask of valid pixels (non-zero depth values)
    valid_pixels = (depth > 0.1).astype(np.uint8)

    # Create a copy of the depth map to store the extrapolated result
    extrapolated_depth = depth.copy()

    # Iterate through dilations and kernels
    for dilation, kernel in zip(dilations, kernels):
        # Dilate the valid pixels mask
        dilated_valid_pixels = cv2.dilate(valid_pixels, kernel, iterations=dilation)

        # Create a mask for the unknown pixels that can be filled in the current iteration
        fill_mask = ((dilated_valid_pixels - valid_pixels) > 0).astype(np.uint8)

        # Apply convolution to the depth map, only considering valid pixels
        # The convolution essentially averages the valid pixels within the kernel
        # and fills the 'fill_mask' regions with these averaged values.
        filtered_depth = cv2.filter2D(extrapolated_depth, -1, kernel, borderType=cv2.BORDER_CONSTANT)

        # Update the extrapolated depth map with the filled pixels
        extrapolated_depth[fill_mask > 0] = filtered_depth[fill_mask > 0]

        # Update the valid pixels mask for the next iteration
        valid_pixels = np.maximum(valid_pixels, dilated_valid_pixels)

        # If all pixels are valid, break the loop
        if np.all(valid_pixels):
            break

    return extrapolated_depth


def fill_in_fast(depth_map, max_depth=100.0,
                 extrapolate=True, blur_type='gaussian'):
    """
    Fast depth map filling: median blur, dilate, and Gaussian blur.
    Modified to directly use the sparse_depth input and return the filled map.
    """
    # Extrapolate sparse depth map
    if extrapolate:
        depth_map = extrapolate_valid_pixels(depth_map)

    # Median blur
    depth_map = cv2.medianBlur(depth_map.astype('float32'), 5)
    depth_map = depth_map.astype('float64')

    # Bilateral or Gaussian blur
    if blur_type == 'bilateral':
        depth_map = cv2.bilateralFilter(depth_map.astype('float32'), 5, 1.5, 2.0)
        depth_map = depth_map.astype('float64')
    elif blur_type == 'gaussian':
        valid_pixels = (depth_map > 0.1)
        blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
        depth_map[valid_pixels] = blurred[valid_pixels]

    # Invert (this seems to be part of a specific post-processing, retaining for now)
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    # Fill zero values if any remain
    mask = (depth_map <= 0.1)
    if np.sum(mask) != 0:
        labeled_array, num_features = label(mask)
        for i in range(num_features):
            index = i + 1
            m = (labeled_array == index)
            m_dilate1 = cv2.dilate(1.0*m, FULL_KERNEL_3, iterations=1)
            m_dilate2 = cv2.dilate(1.0*m_dilate1, FULL_KERNEL_5, iterations=1)
            m_dilate3 = cv2.dilate(1.0*m_dilate2, FULL_KERNEL_7, iterations=1)
            m_dilate4 = cv2.dilate(1.0*m_dilate3, FULL_KERNEL_9, iterations=1)

            depth_map_copy = np.copy(depth_map)
            depth_map_copy[m] = 0.0

            # Ensure there are valid pixels in m_dilate4 > 0 to average from
            if np.sum(m_dilate4 > 0) > 0:
                depth_map[m] = depth_map_copy[m_dilate4 > 0].mean()
            else:
                # If no valid pixels to average from, set to 0 or handle as needed
                depth_map[m] = 0.0

    # Invert back
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    return depth_map