import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from itertools import combinations

def compute_hyperplanes(cluster_centers):
    """
    Computes bisector hyperplanes for each pair of cluster centers.

    Parameters:
    - cluster_centers: Tensor of shape (num_clusters, 2), coordinates of cluster centers.

    Returns:
    - slopes: List of slopes of the bisector hyperplanes.
    - intercepts: List of intercepts of the bisector hyperplanes.
    """
    slopes = []
    intercepts = []
    num_clusters = cluster_centers.shape[0]

    for i, j in combinations(range(num_clusters), 2):
        c1, c2 = cluster_centers[i], cluster_centers[j]

        # Midpoint of the two cluster centers
        midpoint = (c1 + c2) / 2.0

        # Normal vector (perpendicular to the line joining c1 and c2)
        direction = c2 - c1
        normal_vector = np.array([-direction[1], direction[0]])

        # Compute slope and intercept
        if normal_vector[0] != 0:
            slope = normal_vector[1] / normal_vector[0]
            intercept = midpoint[1] - slope * midpoint[0]
        else:
            slope = None  # Vertical line
            intercept = midpoint[0]

        slopes.append(slope)
        intercepts.append(intercept)

    return slopes, intercepts

def partition_plane_with_hyperplanes(cluster_centers, grid_size=300):
    """
    Partitions a plane based on Voronoi-like boundaries and overlays hyperplanes.

    Parameters:
    - cluster_centers: Numpy array of shape (num_clusters, 2), coordinates of cluster centers.
    - grid_size: Resolution of the partition visualization.

    Returns:
    - A visualization of the partitioned plane with hyperplanes.
    """
    cluster_centers = tf.convert_to_tensor(cluster_centers, dtype=tf.float32)

    # Define grid boundaries
    x_min, x_max = np.min(cluster_centers[:, 0]) - 2, np.max(cluster_centers[:, 0]) + 2
    y_min, y_max = np.min(cluster_centers[:, 1]) - 2, np.max(cluster_centers[:, 1]) + 2

    # Create a mesh grid
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points_tf = tf.convert_to_tensor(grid_points, dtype=tf.float32)

    # Assign each grid point to the closest cluster center
    centers_expanded = tf.expand_dims(cluster_centers, axis=0)  # Shape (1, num_clusters, 2)
    distances = tf.norm(grid_points_tf[:, None, :] - centers_expanded, axis=-1)  # (num_grid_points, num_clusters)
    region_assignments = tf.argmin(distances, axis=1)  # Assign to nearest center

    # Reshape assignments for plotting
    region_assignments = region_assignments.numpy().reshape(grid_size, grid_size)

    # Compute hyperplanes
    slopes, intercepts = compute_hyperplanes(cluster_centers.numpy())

    # Plot the partitioned regions
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, region_assignments, alpha=0.3, cmap='viridis')  # Background partition
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], color="red", marker="x", s=100, label="Cluster Centers")
    '''
    # Plot the hyperplanes
    x_vals = np.linspace(x_min, x_max, 100)
    for slope, intercept in zip(slopes, intercepts):
        if slope is not None:
            y_vals = slope * x_vals + intercept
            plt.plot(x_vals, y_vals, 'r--', linewidth=1, label="Bisector Hyperplane")
        else:  # Vertical line
            plt.axvline(intercept, color='r', linestyle='--', linewidth=1, label="Vertical Hyperplane")
    '''
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    #plt.legend()
    plt.title("Cluster Partitioning with Hyperplanes")
    plt.grid()
    plt.savefig("img/partition_with_hyperplanes.png")

def initialize_grid(n_voxels_x=20, n_voxels_y=20):
    _x = tf.linspace(0., 1., n_voxels_x)
    _y = tf.linspace(0., 1., n_voxels_y)
    X, Y = tf.meshgrid(_x, _y)

    # Flatten and convert to trainable variables
    X = tf.Variable(tf.reshape(X, [-1]), trainable=True)
    Y = tf.Variable(tf.reshape(Y, [-1]), trainable=True)

    # Initialize trainable voxel sizes
    dX = tf.Variable(tf.ones_like(X), trainable=True)
    dY = tf.Variable(tf.ones_like(Y), trainable=True)

    return tf.stack([X, Y], axis=1)

cluster_centers = initialize_grid()
partition_plane_with_hyperplanes(cluster_centers)
