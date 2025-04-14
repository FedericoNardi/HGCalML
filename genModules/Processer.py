import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Some helper functions
# Generate 3D coordinates for each voxel
def generate_grid(grid_size):
    x = -300+600*tf.range(grid_size[0], dtype=tf.float32)/grid_size[0]
    y = -300+600*tf.range(grid_size[1], dtype=tf.float32)/grid_size[1]
    z = 600*tf.range(grid_size[2], dtype=tf.float32)/grid_size[2]
    grid_x, grid_y, grid_z = tf.meshgrid(x, y, z, indexing='ij')
    return tf.stack([grid_x, grid_y, grid_z], axis=-1)  # Shape: (X, Y, Z, 2)

# Calculate distances between centroids and deposit coordinates
def calculate_distances(centroids, deposits):
    # Expand dimensions to allow broadcasting
    centroids = tf.expand_dims(centroids, axis=1)  # Shape: (num_centroids, 1, 2)
    deposits = tf.expand_dims(deposits, axis=0)  # Shape: (1, num_deposits, 2)
    # Calculate distances
    distances = tf.norm(centroids - deposits, axis=-1)  # Shape: (num_centroids, num_deposits)
    return distances

# Assign each deposit to the closest centroid, and calculate the total energy for each centroid
def assign_deposits_to_centroids(centroids, deposits, energies):
    deposits = tf.cast(deposits, tf.float32)
    distances = calculate_distances(centroids, deposits)  # Shape: (num_centroids, num_deposits)
    # Find the closest centroid for each deposit
    closest_centroid = tf.argmin(distances, axis=0)  # Shape: (num_deposits,)
    # Calculate the total energy for each centroid
    total_energy = tf.math.unsorted_segment_sum(energies, closest_centroid, num_segments=tf.shape(centroids)[0])
    return total_energy

# 3D Visualization
def plot_3d_voronoi(centroids, deposits, energies, fname='img/3d_voronoi.png'):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    mask = energies > 0.1
    scatter = ax.scatter(deposits[:, 0][mask], deposits[:, 1][mask], deposits[:, 2][mask], c=energies[mask], cmap='viridis', s=1, alpha=0.05)
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='red', marker='o', s=5)
    plt.colorbar(scatter, label='Energy [GeV]')
    plt.title('3D Energy Deposits and Centroids')
    plt.savefig(fname)

def voronoi_assignment(centroids, resolution=100):
    grid = generate_grid((resolution, resolution, resolution))  # Shape: (resolution, resolution, 2)
    grid = tf.reshape(grid, (-1, 3))  # Shape: (resolution*resolution, 2)
    grid = tf.cast(grid, tf.float32)
    distances = calculate_distances(centroids, grid)  # Shape: (num_centroids, num_deposits)
    # Find the closest centroid for each deposit
    closest_centroid = tf.argmin(distances, axis=0)  # Shape: (num_deposits,)
    return grid, closest_centroid

def estimate_voronoi_volumes_3d(points, n_samples=100000, bounds=None):
    """
    Estimate the volume of 3D Voronoi regions using Monte Carlo sampling.

    Args:
        points: Tensor of shape [N, 3] — the Voronoi seed points.
        n_samples: Number of Monte Carlo samples.
        bounds: List of tuples [(xmin, xmax), (ymin, ymax), (zmin, zmax)] — bounding box.

    Returns:
        volumes: Tensor of shape [N,] with estimated volumes.
    """
    N = tf.shape(points)[0]

    if bounds is None:
        # Auto bounding box from point cloud
        min_vals = tf.reduce_min(points, axis=0)
        max_vals = tf.reduce_max(points, axis=0)
        bounds = list(zip(min_vals.numpy(), max_vals.numpy()))
    
    # Sample uniformly within bounds
    samples = tf.stack([
        tf.random.uniform((n_samples,), minval=low, maxval=high)
        for (low, high) in bounds
    ], axis=-1)  # [n_samples, 3]

    # Compute distances to all points
    samples_exp = tf.expand_dims(samples, axis=1)      # [n_samples, 1, 3]
    points_exp = tf.expand_dims(points, axis=0)        # [1, N, 3]
    dists = tf.norm(samples_exp - points_exp, axis=-1) # [n_samples, N]

    # Find the nearest point for each sample
    nearest = tf.cast(tf.argmin(dists, axis=1), tf.int32)                 # [n_samples]

    # Count number of samples per region
    region_counts = tf.math.bincount(nearest, minlength=N, maxlength=N, dtype=tf.float32)

    # Scale by total volume of bounding box
    total_volume = tf.reduce_prod(
        tf.constant([high - low for (low, high) in bounds], dtype=tf.float32)
    )

    volumes = region_counts / tf.cast(n_samples, tf.float32) * total_volume
    return volumes

# Wrapper class starts here

class Voronoi3D():
    def __init__(self, resolution=100):
        self.resolution = resolution
        
    def __call__(self, centroids, event_features, return_volumes=False):
        '''
        centroids: Tensor of shape (num_centroids, 3)
        event_features: Tensor of shape (num_deposits, 4) [coordinates, energy]
        '''
        self.volumes = estimate_voronoi_volumes_3d(centroids)
        if (return_volumes):
            return assign_deposits_to_centroids(centroids, event_features[:, :3], event_features[: ,3]), self.volumes
        return assign_deposits_to_centroids(centroids, event_features[:, :3], event_features[: ,3])

    def plot(self, centroids, event_features):
        grid, closest_centroid = voronoi_assignment(centroids, resolution=self.resolution)
        assigned_energies = tf.gather(event_features[:, 3], closest_centroid)
        plot_3d_voronoi(centroids, grid, assigned_energies)