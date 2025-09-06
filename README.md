## 3D Box Detection and Pose Estimation

# Overview
This project implements a computer vision pipeline for detecting and estimating the 6DOF pose of rectangular boxes in 3D space using RGB-D camera data. The system processes depth maps and color images to generate 3D point clouds, segments box-like objects, and estimates their pose using principal component analysis (PCA).

# Features
- 3D Point Cloud Generation: Converts depth maps to 3D point clouds using camera intrinsic parameters
- Box Segmentation: Uses DBSCAN clustering combined with shape analysis to identify box-like objects
- Pose Estimation: Employs PCA to determine the 6DOF pose (position and orientation) of detected boxes
- Visualization: Provides 3D visualizations with coordinate frames showing box orientation

# Dependencies
NumPy
Matplotlib
Plotly
Scikit-learn
OpenCV
SciPy (implicit via sklearn)

# Use Cases
Robotic manipulation and grasping
Augmented reality object placement
3D scene understanding
Industrial automation and quality control
This implementation provides a robust foundation for 3D object detection in structured environments where rectangular objects need to be precisely located and oriented.
