from dataloaders.modified_utils import *

from matplotlib import pyplot as plt
# === Paths ===
base_path = r"D:\spa\SFA3D\dataset\kitti\testing"
calib_file = os.path.join(base_path, "calib", "000000.txt")
velo_file = os.path.join(base_path, "velodyne", "000000.bin")
image_file = os.path.join(base_path, "image_2", "000000.png")

# === Load calibration ===
calib = read_calib_file(calib_file)

# Ensure required keys exist
if not all(k in calib for k in ["P2", "Tr_velo_to_cam", "R0_rect"]):
    raise ValueError("Calibration file missing required keys: P2, Tr_velo_to_cam, or R0_rect")

P2 = calib["P2"].reshape(3, 4)
Tr_velo_to_cam = calib["Tr_velo_to_cam"].reshape(3, 4)
R0_rect = calib["R0_rect"].reshape(3, 3)

# === Load velodyne point cloud ===
velo_points = read_depth(velo_file)  # Will return (N, 4)

# === Load image to get dimensions ===
import cv2
image = cv2.imread(image_file)
H, W = image.shape[:2]

# === Project velodyne to depth map ===
depth_map = project_velodyne_to_depth_map(
    velo_points,
    P_rect_xx=P2,
    Tr_velo_to_cam=Tr_velo_to_cam,
    R0_rect=R0_rect,
    img_height=H,
    img_width=W,
    max_depth=80.0
)
print(depth_map.shape)
# === Visualize ===
plt.figure(figsize=(10, 5))
plt.imshow(depth_map.squeeze(), cmap='plasma')
plt.title("Projected Sparse Depth Map from LiDAR")
plt.colorbar(label='Depth (m)')
plt.axis('off')
plt.show()
