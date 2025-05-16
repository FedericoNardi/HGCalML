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

def estimate_voronoi_volumes_3d(points, n_samples=100000, bounds=None, seed=42):
    """
    Estimate the volume of 3D Voronoi regions using Monte Carlo sampling.

    Args:
        points: Tensor of shape [N, 3] — the Voronoi seed points (in mm).
        n_samples: Number of Monte Carlo samples.
        bounds: List of tuples [(xmin, xmax), (ymin, ymax), (zmin, zmax)] in mm.
        seed: Integer seed for deterministic sampling.

    Returns:
        volumes: Tensor of shape [N,] with estimated volumes in mm³.
    """
    tf.random.set_seed(seed)  # Ensures determinism

    N = tf.shape(points)[0]

    if bounds is None:
        # Auto bounding box from point cloud
        min_vals = tf.reduce_min(points, axis=0)
        max_vals = tf.reduce_max(points, axis=0)
        bounds = list(zip(min_vals.numpy(), max_vals.numpy()))

    # Sample uniformly within bounds (in mm)
    samples = tf.stack([
        tf.random.uniform((n_samples,), minval=b[0], maxval=b[1], dtype=tf.float32)
        for b in bounds
    ], axis=-1)  # Shape: (n_samples, 3)

    # Compute distances to centroids
    dists = tf.norm(tf.expand_dims(points, axis=1) - tf.expand_dims(samples, axis=0), axis=-1)  # [N, n_samples]
    closest = tf.argmin(dists, axis=0)  # [n_samples]

    # Count assignments per centroid
    counts = tf.math.unsorted_segment_sum(
        data=tf.ones_like(closest, dtype=tf.float32),
        segment_ids=closest,
        num_segments=N
    )

    # Volume of bounding box (in mm³)
    box_volume = tf.reduce_prod([b[1] - b[0] for b in bounds])

    # Estimate volume per centroid
    volumes = counts / tf.cast(n_samples, tf.float32) * box_volume

    return volumes  # Units: mm³

# Wrapper class starts here

class Voronoi3D:
    def __call__(self, centroids: tf.Tensor, voxels: tf.Tensor):

        N = tf.shape(centroids)[0]
        M = tf.shape(voxels)[0]
        points = voxels[:, :3]   # (M, 3)
        energies = voxels[:, 3]  # (M,)
        
        # Compute pairwise squared distances between centroids and voxel positions
        c_sq = tf.reduce_sum(tf.square(centroids), axis=-1, keepdims=True)  # (N, 1)
        p_sq = tf.reduce_sum(tf.square(points), axis=-1, keepdims=True)     # (M, 1)
        dot = tf.matmul(centroids, points, transpose_b=True)                # (N, M)
        
        # ||c - p||^2 = ||c||^2 - 2c⋅p + ||p||^2
        dists = tf.transpose(c_sq - 2.0 * dot + tf.transpose(p_sq, perm=[1, 0]))  # (M, N)
        
        # Find the closest centroid to each voxel
        closest = tf.argmin(dists, axis=-1, output_type=tf.int32)  # (M,)
        
        # Sum energies assigned to each centroid
        energy_sum = tf.math.unsorted_segment_sum(energies, closest, num_segments=N)

        # Volume estimation using MC sampling (graph-safe)
        def estimate_volume(c):
            n_samples = 20000
    
            # Compute bounds on-the-fly
            min_vals = tf.reduce_min(c, axis=0)
            max_vals = tf.reduce_max(c, axis=0)
    
            samples = tf.stack([
                tf.random.uniform((n_samples,), min_vals[0], max_vals[0]),
                tf.random.uniform((n_samples,), min_vals[1], max_vals[1]),
                tf.random.uniform((n_samples,), min_vals[2], max_vals[2]),
            ], axis=-1)  # (n_samples, 3)
    
            c_sq = tf.reduce_sum(tf.square(c), axis=1, keepdims=True)          # (N, 1)
            s_sq = tf.reduce_sum(tf.square(samples), axis=1, keepdims=True)    # (S, 1)
            dot = tf.matmul(samples, c, transpose_b=True)                      # (S, N)
            dist = s_sq - 2 * dot + tf.transpose(c_sq)                         # (S, N)
    
            nearest = tf.argmin(dist, axis=1, output_type=tf.int32)            # (S,)
            volume_counts = tf.math.bincount(nearest, minlength=tf.shape(c)[0], maxlength=tf.shape(c)[0], dtype=tf.float32)
    
            total_volume = tf.reduce_prod(max_vals - min_vals)
            estimated_volume = (volume_counts / tf.cast(n_samples, tf.float32)) * total_volume
            return estimated_volume  # (N,)
    
        volumes = estimate_volume(centroids)  # (N,)
    
        return energy_sum, volumes
        # return assign_deposits_to_centroids(centroids, event_features[:, :3], event_features[: ,3])

    def plot(self, centroids, event_features):
        grid, closest_centroid = voronoi_assignment(centroids, resolution=self.resolution)
        assigned_energies = tf.gather(event_features[:, 3], closest_centroid)
        plot_3d_voronoi(centroids, grid, assigned_energies)