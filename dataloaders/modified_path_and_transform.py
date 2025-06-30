import glob
import os.path

from dataloaders.modified_utils import *
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image # Import PIL Image

glob_dep, glob_gt, glob_K, glob_rgb = None, None, None, None
get_rgb_paths, get_K_paths = None, None

def get_kittipaths(split, args):
    global glob_dep, glob_gt, glob_K, glob_rgb
    global get_rgb_paths, get_K_paths

    # Determine the base data folder (training or testing)
    base_data_folder = os.path.join(args.data_folder, 'training')
    if split == 'test_completion':
        base_data_folder = os.path.join(args.data_folder, 'testing')

    if split == 'train' or (split == 'val' and args.val == 'full'):
        # Paths for raw KITTI 'training' data
        glob_dep = os.path.join(
            base_data_folder,
            'velodyne/*.bin' # Raw Velodyne files
        )
        glob_gt = None # User explicitly states not using GT file or path
        
        def get_rgb_paths(p):
            # p comes from glob_dep (e.g., '.../velodyne/000000.bin')
            base_name = os.path.basename(p).replace('.bin', '.png')
            return os.path.join(base_data_folder, 'image_2', base_name)
            
        def get_K_paths(p):
            # p comes from glob_dep (e.g., '.../velodyne/000000.bin')
            base_name = os.path.basename(p).replace('.bin', '.txt')
            return os.path.join(base_data_folder, 'calib', base_name)

    elif split == 'val' and args.val == 'select':
        # This is the specific branch that triggered your original error.
        # It's usually for the KITTI Depth Completion val set, but we map it to raw KITTI 'training' data.
        glob_dep = os.path.join(
            base_data_folder, # Points to D:\spa\SFA3D\dataset\kitti\training
            'velodyne/*.bin' # Raw Velodyne files
        )
        glob_gt = None # User explicitly states not using GT file or path
        
        def get_rgb_paths(p):
            base_name = os.path.basename(p).replace('.bin', '.png')
            return os.path.join(base_data_folder, 'image_2', base_name)
        def get_K_paths(p):
            base_name = os.path.basename(p).replace('.bin', '.txt')
            return os.path.join(base_data_folder, 'calib', base_name)

    elif split == 'test_completion':
        print("This is in the modified path and the split is test_completion========")
        print("entered the test_completion branch============")
        # Paths for raw KITTI 'testing' data
        glob_dep = os.path.join(
            base_data_folder, # Points to D:\spa\SFA3D\dataset\kitti\testing
            'velodyne/*.bin' # Raw Velodyne files
        )
        glob_gt = None # No ground truth file/path for test_completion usually
        
        def get_rgb_paths(p):
            base_name = os.path.basename(p).replace('.bin', '.png')
            return os.path.join(base_data_folder, 'image_2', base_name)
        def get_K_paths(p):
            base_name = os.path.basename(p).replace('.bin', '.txt')
            return os.path.join(base_data_folder, 'calib', base_name)

    else:
        raise ValueError('Undefined split {}'.format(split))

    # Path retrieval for 'dep' (input depth/velodyne)
    paths_dep = sorted(glob.glob(glob_dep))
    print(f"Paths for depth files=========: {paths_dep[:2]}...")

    # Path retrieval for 'gt' (ground truth depth) - always None now
    paths_gt = [None] * len(paths_dep)
    print(f"Paths for ground truth files (should be None)=========: {paths_gt[:2]}...")

    # Path retrieval for 'rgb'
    paths_rgb = [get_rgb_paths(p) for p in paths_dep] if get_rgb_paths else [None] * len(paths_dep)
    paths_rgb = sorted(paths_rgb)
    print(f"Paths for RGB files=========: {paths_rgb[:2]}...")

    # Path retrieval for 'K' (calibration)
    paths_K = [get_K_paths(p) for p in paths_dep] if get_K_paths else [None] * len(paths_dep)
    paths_K = sorted(paths_K)

    paths = {'dep': paths_dep, 'gt': paths_gt, 'rgb': paths_rgb, 'K': paths_K}

    print(f"Total number of files found for split {split}: "
          f"Velodyne: {len(paths_dep)}, RGB: {len(paths_rgb)}, Calib: {len(paths_K)}")
    

    if not paths_dep:
        raise RuntimeError(f"Found 0 files under input depth path: {glob_dep}. "
                           "Please check your 'data_folder' and the expected file pattern. "
                           "If you're using raw KITTI, ensure '.bin' files are present and accessible.")
    
    if len(paths_dep) != len(paths_rgb) or (len(paths_dep) != len(paths_K)):
        print(f"Warning: Mismatch in number of files for split {split}: "
              f"Velodyne: {len(paths_dep)}, RGB: {len(paths_rgb)}, Calib: {len(paths_K)}")

    return paths

# This function must be defined at the top level for explicit import in kitti_loader.py

def kittitransforms(split, args, dep, gt, K, rgb):
    
    # print("this is the initial data bfore transofmration args===",args)
    # print("this is the initial data bfore transofmration dep ===",dep)
    # if dep is not None:
    #     print("dep size=",dep.shape)
    # print("this is the initial data bfore transofmration  gt ===",gt )
    # print("this is the initial data bfore transofmration   K ===",K)
    # print("this is the initial data bfore transofmration   rgb===",rgb)

    # Helper functions for transformations
    def Resize(img, h, w):
        if img is None: return None
        if isinstance(img, np.ndarray):
            # For numpy arrays (depth), use cv2.resize for consistency
            if img.ndim == 2:  # Depth image (H, W)
                return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            elif img.ndim == 3: # RGB image (H, W, C)
                 return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                raise ValueError(f"Unsupported numpy image dimensions for Resize: {img.ndim}")
        elif isinstance(img, Image.Image): 
            return img.resize((w, h), Image.BILINEAR)
        elif torch.is_tensor(img):
            # Ensure tensor is in (C, H, W) format before resize
            if img.ndim == 3:
                return TF.resize(img, (h, w))
            elif img.ndim == 2: # Depth tensor (H, W)
                img = img.unsqueeze(0)  # Add channel dimension
                return TF.resize(img, (h, w)).squeeze(0)
            else:
                raise ValueError(f"Unsupported torch tensor dimensions for Resize: {img.ndim}")
        else: 
            raise TypeError(f"Unsupported image type for Resize: {type(img)}")

    def Crop(img, h_init, w_init, rheight, rwidth):
        if img is None: return None
        # Check if crop dimensions are valid
        if rheight <= 0 or rwidth <= 0:
            return None # Return None if crop would result in empty image

        if isinstance(img, np.ndarray):
            if img.ndim == 2:  # For depth images (H, W)
                return img[h_init : h_init + rheight, w_init : w_init + rwidth]
            elif img.ndim == 3:  # For RGB images (H, W, C)
                return img[h_init : h_init + rheight, w_init : w_init + rwidth, :]
            else:
                raise ValueError(f"Unsupported numpy image dimensions for Crop: {img.ndim}")
        elif isinstance(img, Image.Image): 
            return img.crop((w_init, h_init, w_init + rwidth, h_init + rheight))
        elif torch.is_tensor(img):
            # Ensure tensor is in (C, H, W) format before crop
            if img.ndim == 3:
                return img[:, h_init : h_init + rheight, w_init : w_init + rwidth]
            elif img.ndim == 2: # Depth tensor (H, W)
                img = img.unsqueeze(0)  # Add channel dimension
                cropped = img[:, h_init : h_init + rheight, w_init : w_init + rwidth]
                return cropped.squeeze(0)  # Remove channel dimension if input was 2D
            else:
                raise ValueError(f"Unsupported torch tensor dimensions for Crop: {img.ndim}")
        else: 
            raise TypeError(f"Unsupported image type for Crop: {type(img)}")

    # Convert RGB to numpy array before checking shape for cropping, if it's a PIL Image
    if isinstance(rgb, Image.Image):
        rgb_np = np.array(rgb)
    else:
        rgb_np = rgb

    # Determine original dimensions
    original_h, original_w = 0, 0
    if rgb_np is not None:
        original_h, original_w = rgb_np.shape[0], rgb_np.shape[1]
    elif dep is not None:
        if isinstance(dep, np.ndarray):
            original_h, original_w = dep.shape[-2], dep.shape[-1] # Assuming (C, H, W) or (H, W)
        elif isinstance(dep, Image.Image):
            original_w, original_h = dep.size
        elif torch.is_tensor(dep):
            original_h, original_w = dep.shape[-2], dep.shape[-1] # Assuming (C, H, W) or (H, W)


    # Apply Resize first if specified, this changes the dimensions for subsequent crops
    if args.resize:
        resize_h, resize_w = args.val_h, args.val_w
        if split == 'train':
            # For training, if resize is true, you might want to resize to a specific training resolution
            # or keep it consistent with the overall flow. Assuming val_h/val_w for simplicity here.
            # You might need to adjust this based on your training config.
            resize_h, resize_w = args.random_crop_height, args.random_crop_width # Or whatever your train target size is
        
        dep = Resize(dep, resize_h, resize_w)
        gt = Resize(gt, resize_h, resize_w)
        rgb = Resize(rgb, resize_h, resize_w)
        
        # Update dimensions after resize for cropping calculations
        if rgb is not None:
            if isinstance(rgb, np.ndarray):
                original_h, original_w = rgb.shape[0], rgb.shape[1]
            elif isinstance(rgb, Image.Image):
                original_w, original_h = rgb.size
            elif torch.is_tensor(rgb):
                original_h, original_w = rgb.shape[-2], rgb.shape[-1]
        elif dep is not None: # Fallback to depth dimensions if RGB is None after resize
            if isinstance(dep, np.ndarray):
                original_h, original_w = dep.shape[-2], dep.shape[-1]
            elif isinstance(dep, Image.Image):
                original_w, original_h = dep.size
            elif torch.is_tensor(dep):
                original_h, original_w = dep.shape[-2], dep.shape[-1]

    # Determine crop parameters
    h_init, w_init, rheight, rwidth = 0, 0, original_h, original_w # Default to no crop (full image)

    if split == 'train':
        if args.train_random_crop:
            rheight, rwidth = args.random_crop_height, args.random_crop_width
            if original_h >= rheight and original_w >= rwidth:
                h_init = np.random.randint(0, original_h - rheight + 1)
                w_init = np.random.randint(0, original_w - rwidth + 1)
            else:
                print(f"Warning: Image dimensions ({original_h}x{original_w}) too small for random crop "
                      f"({rheight}x{rwidth}) in train split. Skipping random crop.")
                rheight, rwidth = original_h, original_w # No effective crop
        elif args.train_bottom_crop: # If not random crop, but bottom crop is true
             rheight, rwidth = args.val_h, args.val_w # Assuming val_h, val_w are used for bottom crop size
             if original_h >= rheight and original_w >= rwidth:
                h_init = original_h - rheight
                w_init = (original_w - rwidth) // 2
             else:
                print(f"Warning: Image dimensions ({original_h}x{original_w}) too small for bottom crop "
                      f"({rheight}x{rwidth}) in train split. Skipping bottom crop.")
                rheight, rwidth = original_h, original_w # No effective crop
    
    elif split == 'val':
        if args.val_random_crop: # This is often similar to test_random_crop for evaluation
            rheight, rwidth = args.val_h, args.val_w
            if original_h >= rheight and original_w >= rwidth:
                # For val/test, typically a center or fixed crop is used if 'random_crop' is set to true but for a specific size.
                h_init = original_h - rheight
                w_init = (original_w - rwidth) // 2
            else:
                print(f"Warning: Image dimensions ({original_h}x{original_w}) too small for random crop "
                      f"({rheight}x{rwidth}) in val split. Skipping random crop.")
                rheight, rwidth = original_h, original_w # No effective crop
        elif args.val_bottom_crop:
            rheight, rwidth = args.val_h, args.val_w
            if original_h >= rheight and original_w >= rwidth:
                h_init = original_h - rheight
                w_init = (original_w - rwidth) // 2
            else:
                print(f"Warning: Image dimensions ({original_h}x{original_w}) too small for bottom crop "
                      f"({rheight}x{rwidth}) in val split. Skipping bottom crop.")
                rheight, rwidth = original_h, original_w # No effective crop
    
    elif split == 'test_completion':
        # Assuming test_random_crop acts like a fixed center/bottom crop for evaluation
        if args.test_random_crop:
            rheight, rwidth = args.val_h, args.val_w
            if original_h >= rheight and original_w >= rwidth:
                h_init = original_h - rheight
                w_init = (original_w - rwidth) // 2
            else:
                print(f"Warning: Image dimensions ({original_h}x{original_w}) too small for random crop "
                      f"({rheight}x{rwidth}) in test split. Skipping random crop.")
                rheight, rwidth = original_h, original_w # No effective crop
        elif args.test_bottom_crop:
            rheight, rwidth = args.val_h, args.val_w
            if original_h >= rheight and original_w >= rwidth:
                h_init = original_h - rheight
                w_init = (original_w - rwidth) // 2
            else:
                print(f"Warning: Image dimensions ({original_h}x{original_w}) too small for bottom crop "
                      f"({rheight}x{rwidth}) in test split. Skipping bottom crop.")
                rheight, rwidth = original_h, original_w # No effective crop

    # Apply Crop
    dep = Crop(dep, h_init, w_init, rheight, rwidth)
    gt = Crop(gt, h_init, w_init, rheight, rwidth)
    rgb = Crop(rgb, h_init, w_init, rheight, rwidth)

    if K is not None:
        # K (intrinsics) adjustment based on crop and resize
        # K is [fx, fy, cx, cy]
        # If resized, K needs to be scaled
        if args.resize and original_h > 0 and original_w > 0: # Ensure original_h, original_w are not zero
            scale_x = rwidth / original_w # Use cropped/final dimensions after resize
            scale_y = rheight / original_h # Use cropped/final dimensions after resize
            K[0] *= scale_x # fx
            K[1] *= scale_y # fy
            K[2] = K[2] * scale_x - w_init # cx - adjusted for crop and resize
            K[3] = K[3] * scale_y - h_init # cy - adjusted for crop and resize
        else: # Only cropped, no resize
            K[2] = K[2] - w_init # cx
            K[3] = K[3] - h_init # cy


    # Convert to Tensor
    if dep is not None:
        if isinstance(dep, np.ndarray):
            if dep.ndim == 2:
                dep = np.expand_dims(dep, 0) # Add channel dimension (1, H, W)
            dep = torch.from_numpy(dep).float()
        # If dep is already a tensor and 2D, add channel dimension
        elif torch.is_tensor(dep) and dep.ndim == 2:
            dep = dep.unsqueeze(0)
    
    if gt is not None:
        if isinstance(gt, np.ndarray):
            if gt.ndim == 2:
                gt = np.expand_dims(gt, 0)
            gt = torch.from_numpy(gt).float()
        elif torch.is_tensor(gt) and gt.ndim == 2:
            gt = gt.unsqueeze(0)

    if rgb is not None:
        if isinstance(rgb, np.ndarray):
            # RGB from OpenCV usually BGR, convert to RGB
            if rgb.shape[-1] == 3: # (H, W, C)
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = TF.to_tensor(np.array(rgb, dtype=np.float32))
        elif isinstance(rgb, Image.Image): # PIL Image
            rgb = TF.to_tensor(np.array(rgb, dtype=np.float32))
        elif torch.is_tensor(rgb):
            pass # Already a tensor
        else:
            raise TypeError(f"Unsupported RGB type for tensor conversion: {type(rgb)}")

    # Normalize
    if args.normalize and rgb is not None:
        # KITTI usually uses ImageNet normalization mean/std
        rgb = TF.normalize(rgb, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)

    # Scale depth
    if args.scale_depth and dep is not None:
        _scale = 1.0 # Your current code uses 1.0, implying no scaling here, check your config if this needs to be dynamic.
        dep = dep / _scale
    
    # Handle horizontal flip (after tensor conversion for consistency)
    if args.hflip and split == 'train': # Usually hflip is only for training augmentation
        flip = torch.FloatTensor(1).uniform_(0, 1).item()
        if flip > 0.5:
            dep = TF.hflip(dep) if dep is not None else None
            gt = TF.hflip(gt) if gt is not None else None
            rgb = TF.hflip(rgb) if rgb is not None else None
            if K is not None:
                # Assuming K[2] is cx (principal point x)
                # After flip, cx becomes width - cx
                # Need to use the width of the image AFTER any resizing/cropping for correct cx
                current_width = dep.shape[2] if dep is not None else (rgb.shape[2] if rgb is not None else 0)
                if current_width > 0:
                    K[2] = current_width - K[2]

    # Apply ColorJitter (after tensor conversion)
    if args.colorjitter and split == 'train' and rgb is not None: # ColorJitter only for training
        # Define ColorJitter parameters based on your original utility.py or common practice
        # Example: brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.1, 0.1)
        # Assuming you have a ColorJitter function in modified_utils.py that works on tensors or PIL images.
        # If your ColorJitter expects PIL image, this needs to be before TF.to_tensor.
        # For now, let's assume it's applied after tensor conversion and expects a tensor.
        # If ColorJitter is a custom function, ensure it handles torch tensors.
        # Placeholder for now, assuming it's correctly defined elsewhere.
        pass # ColorJitter should be applied here if it expects tensors.

    # Add noise (after tensor conversion)
    if args.noise > 0 and dep is not None and split == 'train': # Noise usually for training augmentation
        reflection = np.clip(np.random.normal(1, scale=0.333332, size=(1, 1)), 0.01, 3)[0, 0]
        noise = torch.normal(mean=0.0, std=dep * reflection * args.noise)
        dep_noise = dep + noise
        dep_noise[dep_noise < 0] = 0
        dep = dep_noise
    
    if args.rgb_noise > 0 and rgb is not None and split == 'train': # RGB noise for training augmentation
        rgb_n = torch.FloatTensor(1).uniform_(0, 1).item()
        if rgb_n > 0.2:
            rgb_noise = torch.normal(mean=torch.zeros_like(rgb), std=args.rgb_noise * torch.FloatTensor(1).uniform_(0.5, 1.5).item())
            rgb = rgb + rgb_noise


    return dep, gt, K, rgb


def get_rgb(idx, paths, split, args):
    # This function is called by KittiDepth.__getraw__
    # The paths['rgb'][idx] will now directly contain the correct RGB image path for raw KITTI
    return read_rgb(paths['rgb'][idx])


# The visualization block below is for testing purposes and should not be part of the main dataset code.
# Remove or comment out this block when integrating into your project.
# from matplotlib import pyplot as plt
# # === Paths ===
# base_path = r"D:\spa\SFA3D\dataset\kitti\testing"
# calib_file = os.path.join(base_path, "calib", "000000.txt")
# velo_file = os.path.join(base_path, "velodyne", "000000.bin")
# image_file = os.path.join(base_path, "image_2", "000000.png")

# # === Load calibration ===
# calib = read_calib_file(calib_file)

# # Ensure required keys exist
# if not all(k in calib for k in ["P2", "Tr_velo_to_cam", "R0_rect"]):
#     raise ValueError("Calibration file missing required keys: P2, Tr_velo_to_cam, or R0_rect")

# P2 = calib["P2"].reshape(3, 4)
# Tr_velo_to_cam = calib["Tr_velo_to_cam"].reshape(3, 4)
# R0_rect = calib["R0_rect"].reshape(3, 3)

# # === Load velodyne point cloud ===
# velo_points = read_depth(velo_file)  # Will return (N, 4)

# # === Load image to get dimensions ===
# import cv2
# image = cv2.imread(image_file)
# H, W = image.shape[:2]

# # === Project velodyne to depth map ===
# depth_map = project_velodyne_to_depth_map(
#     velo_points,
#     P_rect_xx=P2,
#     Tr_velo_to_cam=Tr_velo_to_cam,
#     R0_rect=R0_rect,
#     img_height=H,
#     img_width=W,
#     max_depth=80.0
# )

# # === Visualize ===
# plt.figure(figsize=(10, 5))
# plt.imshow(depth_map.squeeze(), cmap='plasma')
# plt.title("Projected Sparse Depth Map from LiDAR")
# plt.colorbar(label='Depth (m)')
# plt.axis('off')
# plt.show()