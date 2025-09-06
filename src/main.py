import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
import cv2
from sklearn.cluster import DBSCAN

import plotly.offline as pyo
# pyo.init_notebook_mode(connected=True)

# Load the provided data files
extrinsics = np.load('src/data/extrinsics.npy')
intrinsics = np.load('src/data/intrinsics.npy') 
depth_map = np.load('src/data/one-box.depth.npdata.npy')
color_image = np.load('src/data/one-box.color.npdata.npy')

#Examine the data shapes and contents
print("Extrinsics shape:", extrinsics.shape)
print("Intrinsics shape:", intrinsics.shape)
print("Depth map shape:", depth_map.shape)
print("Color image shape:", color_image.shape)

# If your image is not already uint8, convert it:
if color_image.dtype != np.uint8:
    color_image = cv2.normalize(color_image, None, 0, 255, cv2.NORM_MINMAX)
    color_image = color_image.astype(np.uint8)

color_image = cv2.equalizeHist(color_image)

# # visualize the color image only
# fig = plt.imshow(color_image, cmap='gray')
# plt.title("Color Image")
# plt.axis('off')
# plt.show()


# Intrisic camera parameters
fx = intrinsics[0, 0]
fy = intrinsics[1, 1]
cx = intrinsics[0, 2]
cy = intrinsics[1, 2]
height, width = depth_map.shape
#print(f"fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}, height: {height}, width: {width}")


# Extinsic camera parameters
R = extrinsics[:3, :3]
t = extrinsics[:3, 3]

# print("Rotation matrix R:\n", R)
# print("Translation vector t:\n", t)


# # we use the intrinsics parameters  and depth map to get 3D point cloud
def depth_to_point_cloud(depth_map, intrinsics):
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    height, width = depth_map.shape
    v, u = np.meshgrid(range(height), range(width), indexing='ij')
    valid_depth = depth_map > 0

    Z = - depth_map[valid_depth]
    X = (u[valid_depth] - cx) * Z / fx
    Y = - (v[valid_depth] - cy) * Z / fy

    points = np.column_stack([X, Y, Z])
    return points, valid_depth

points, valid_depth = depth_to_point_cloud(depth_map, intrinsics)
colors_image = color_image[valid_depth]

# fig = go.Figure(data=[go.Scatter3d(
#     x=points[::50, 0], 
#     y=points[::50, 1], 
#     z=points[::50, 2],
#     mode='markers',
#     marker=dict(
#         size=1,
#         color_image=color_image[::50],
#         colorscale='Gray',
#         opacity=0.8
#     )
# )])
# fig.update_layout(title='3D Point Cloud')
# fig.show(renderer="browser")  

def segment_box_simple(points_3d):
    """
    Simple shape-based box detection
    """
    # Sample points if too many
    work_points = points_3d.copy()
    if len(work_points) > 12000:
        indices = np.random.choice(len(work_points), 12000, replace=False)
        work_points = work_points[indices]
    
    # Cluster all points
    clustering = DBSCAN(eps=0.1, min_samples=80).fit(work_points)
    labels = clustering.labels_
    
    unique_labels = np.unique(labels[labels != -1])
    
    best_cluster = None
    best_rectangularity = 0
    
    for label in unique_labels:
        cluster_points = work_points[labels == label]
        
        if len(cluster_points) < 200:  # Skip small clusters
            continue
        
        # Check rectangularity using PCA
        pca = PCA(n_components=3)
        centered = cluster_points - np.mean(cluster_points, axis=0)
        pca.fit(centered)
        
        # Transform to principal components
        transformed = pca.transform(centered)
        
        # Check if it forms a good rectangular shape
        # Good boxes have two dominant dimensions and one small one
        eigenvals = pca.explained_variance_ratio_
        
        # We want: large, medium, small eigenvalues (like a box)
        if eigenvals[2] < 0.1 and eigenvals[1] > 0.15:  # Flat but not too flat
            rectangularity = eigenvals[1] * (1 - eigenvals[2])
            
            if rectangularity > best_rectangularity:
                best_rectangularity = rectangularity
                best_cluster = cluster_points
                
        print(f"Cluster {label}: {len(cluster_points)} points, eigenvals: {eigenvals}, rectangularity: {rectangularity:.3f}")
    
    if best_cluster is not None:
        print(f"Found box with rectangularity score: {best_rectangularity:.3f}")
        return best_cluster, labels, work_points
    else:
        # Fallback to largest reasonable cluster
        counts = [np.sum(labels == label) for label in unique_labels]
        largest_cluster = unique_labels[np.argmax(counts)]
        return work_points[labels == largest_cluster], labels, work_points

box_points, labels, close_points = segment_box_simple(points)
print(f"Segmented {box_points.shape[0]} points belonging to the box.")

# # Visualize the segmented box points
# fig = go.Figure(data=[go.Scatter3d(
#     x=close_points[::1, 0],
#     y=close_points[::1, 1],
#     z=close_points[::1, 2],
#     mode='markers',
#     marker=dict(
#         size=1,
#         color='gray',
#         opacity=0.9
#     )
# )])                
# fig.add_trace(go.Scatter3d(
#     x=box_points[:, 0],
#     y=box_points[:, 1],
#     z=box_points[:, 2],
#     mode='markers',
#     marker=dict(
#         size=1,
#         color='green',
#         opacity=0.9
#     )
# ))
# fig.update_layout(title='Segmented Box Points (green) in 3D Point Cloud (gray)')
# fig.show(renderer="browser")

# we estimate the pose of the box using pca
def estimate_box_pose_pca(box_points):
    """
    Force Z-axis to be the surface normal (minimum variance direction)
    """
    centroid = np.mean(box_points, axis=0)
    centered_points = box_points - centroid
    
    pca = PCA(n_components=3)
    pca.fit(centered_points)
    
    components = pca.components_
    variances = pca.explained_variance_ratio_

    z_axis = components[2]
    # Make sure Z points up (positive Z in camera frame)
    if z_axis[2] < 0:
        z_axis = -z_axis
    
    # Choose X as the direction with maximum variance in the plane perpendicular to Z
    x_axis = components[0]
    
    # Ensure X is perpendicular to Z
    x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # Y = Z × X (right-handed system)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # Build rotation matrix: [X Y Z]
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = rotation_matrix
    pose_matrix[:3, 3] = centroid
    
    print(f"Variances: {variances}")
    print(f"Surface normal (Z): {z_axis}")
    print(f"Box axes - X: {x_axis}, Y: {y_axis}")
    
    return pose_matrix, centroid, rotation_matrix
pose_matrix, centroid, rotation_matrix = estimate_box_pose_pca(box_points)
print("Estimated Centroid:", centroid)
print("Principal Axes:\n", pose_matrix)
print("Rotation Matrix:\n", rotation_matrix)



# Visualize with corrected axes
fig = go.Figure()

# Plot all points (gray)
fig.add_trace(go.Scatter3d(
    x=close_points[:, 0], 
    y=close_points[:, 1],
    z=close_points[:, 2],
    mode='markers',
    marker=dict(size=1, color='gray', opacity=0.9),
    name='All points'
))

# Plot box points (green)
fig.add_trace(go.Scatter3d(
    x=box_points[:, 0],
    y=box_points[:, 1],
    z=box_points[:, 2],
    mode='markers',
    marker=dict(size=2, color='green', opacity=0.9),
    name='Box points'
))

# Draw coordinate frame axes
axis_length = 0.15
origin = centroid

# X-axis (RED) - along box length
x_end = origin + rotation_matrix[:, 0] * axis_length
fig.add_trace(go.Scatter3d(
    x=[origin[0], x_end[0]], y=[origin[1], x_end[1]], z=[origin[2], x_end[2]],
    mode='lines+text', line=dict(color='red', width=8),
    text=['', 'X'], textposition='middle right', name='X-axis (Length)'
))

# Y-axis (GREEN) - along box width  
y_end = origin + rotation_matrix[:, 1] * axis_length
fig.add_trace(go.Scatter3d(
    x=[origin[0], y_end[0]], y=[origin[1], y_end[1]], z=[origin[2], y_end[2]],
    mode='lines+text', line=dict(color='green', width=8),
    text=['', 'Y'], textposition='middle right', name='Y-axis (Width)'
))

# Z-axis (BLUE) - normal to surface
z_end = origin + rotation_matrix[:, 2] * axis_length  
fig.add_trace(go.Scatter3d(
    x=[origin[0], z_end[0]], y=[origin[1], z_end[1]], z=[origin[2], z_end[2]],
    mode='lines+text', line=dict(color='blue', width=8),
    text=['', 'Z'], textposition='middle right', name='Z-axis (Normal)'
))

fig.update_layout(title='Segmented Box Points (green) in 3D Point Cloud (gray)')
fig.show(renderer="browser")
