import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the provided data files
extrinsics = np.load('src/data/extrinsics.npy')
intrinsics = np.load('src/data/intrinsics.npy') 
depth_map = np.load('src/data/one-box.depth.npdata.npy')
color_image = np.load('src/data/one-box.color.npdata.npy')

# Examine the data shapes and contents
# print("Extrinsics shape:", extrinsics.shape)
# print("Intrinsics shape:", intrinsics.shape)
# print("Depth map shape:", depth_map.shape)
# print("Color image shape:", color_image.shape)


# Intrisic camera parameters
fx = intrinsics[0, 0]
fy = intrinsics[1, 1]
cx = intrinsics[0, 2]
cy = intrinsics[1, 2]

height, width = depth_map.shape
print(f"fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}, height: {height}, width: {width}")


# Extinsic camera parameters
R = extrinsics[:3, :3]
t = extrinsics[:3, 3]
print("Rotation matrix R:\n", R)
print("Translation vector t:\n", t)


# we use the intrinsics parameters  and depth map to get 3D point cloud
points=[]

for v in range(height):
    for u in range(width):
        Z = depth_map[v, u]
        if Z == 0:  # we skip the points with zero depth
            continue
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        points.append([X, Y, Z])


# visualize the 3D point cloud
points = np.array(points)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='.')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()


