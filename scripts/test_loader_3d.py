import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import glob
import os

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import uproot

def randomizer(x, y, z):
    '''
    Implementing a 3D rotation matrix for phi in [-30, 30]deg and \theta in [0, 360]deg around a pole in 0. phi is the angle in the zx plane, theta in the yz plane.
    '''
    # Randomize the angles
    phi = tf.random.uniform((), minval=-30, maxval=30, dtype=tf.float32)
    theta = tf.random.uniform((), minval=0, maxval=45, dtype=tf.float32)
    # Convert to radians
    phi = phi * np.pi / 180
    theta = theta * np.pi / 180
    # Define the rotation matrices
    R_phi = tf.convert_to_tensor([[tf.cos(phi), 0, tf.sin(phi)], [0, 1, 0], [-tf.sin(phi), 0, tf.cos(phi)]], dtype=tf.float32)
    R_theta = tf.convert_to_tensor([[1, 0, 0], [0, tf.cos(theta), -tf.sin(theta)], [0, tf.sin(theta), tf.cos(theta)]], dtype=tf.float32)
    # Rotate the coordinates
    x_rot, y_rot, z_rot = tf.tensordot(R_phi, tf.tensordot(R_theta, tf.stack([x, y, z], axis=0), axes=1), axes=1)
    return x_rot, y_rot, z_rot


# single-file function 
def load_data_unit():
    with uproot.open("/media/disk/g4_showers/unif/photon_showers_96.root:photon_sim") as file:
        evt_id = file['EventID'].array(library='np')
        hit_x = file['x'].array(library='np') # mm
        hit_y = file['y'].array(library='np') # mm
        hit_z = file['z'].array(library='np')+300 # mm
        hit_e = file['dE'].array(library='np')*1e-3 # GeV  
        primary_E = file['primaryE'].array(library='np')*1e-3 # GeV  
    print(f'max x: {np.max(hit_x)}, min x: {np.min(hit_x)}')
    print(f'max y: {np.max(hit_y)}, min y: {np.min(hit_y)}')
    print(f'max z: {np.max(hit_z)}, min z: {np.min(hit_z)}')
    return evt_id, hit_x, hit_y, hit_z, hit_e, primary_E


# Create a dataset from the data, where each entry corresponds to a different event id: new event when ID changes

def create_dataset(IDs, x, y, z, dE, prim_E):
    dataset = []
    for i in tqdm(range(len(np.unique(IDs))), desc='Creating dataset: '):
        mask = IDs==i
        # Rotate the coordinates
        x_rot, y_rot, z_rot = randomizer(tf.cast(x[mask], tf.float32), tf.cast(y[mask], tf.float32), tf.cast(z[mask], tf.float32))
        dataset.append(np.stack([x_rot, y_rot, z_rot, dE[mask], prim_E[mask]], axis=-1))
    return dataset

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
    total_energy = tf.math.unsorted_segment_sum(energies, closest_centroid, num_segments=len(centroids))
    return total_energy

# Load the data
evt_id, hit_x, hit_y, hit_z, hit_e, primary_E = load_data_unit()
# Create the dataset
dataset = create_dataset(evt_id, hit_x, hit_y, hit_z, hit_e, primary_E)

# Example centroid coordinates
num_centroids = 250

centroids = tf.random.uniform((num_centroids, 3), minval=[-300, -300, 0], maxval=[300,300,600], dtype=tf.float32)
event = dataset[50]
print(f'Max x: {np.max(event[:,0])}, Min x: {np.min(event[:,0])}')
print(f'Max y: {np.max(event[:,1])}, Min y: {np.min(event[:,1])}')
print(f'Max z: {np.max(event[:,2])}, Min z: {np.min(event[:,2])}')

coordinates = tf.stack([event[:,0], event[:,1], event[:,2]], axis=-1)
energies = event[:,3]

# Plot 2D histogram of the energy deposits
plt.figure(figsize=(20, 8))
plt.subplot(121)
plt.hist2d(event[:,0], event[:,2], bins=100, weights=event[:,3], cmap='viridis')
plt.colorbar(label='Energy [GeV]')  # Add a colorbar
plt.scatter(centroids[:, 0], centroids[:, 2], c='red', marker='o', s=50)
plt.xlabel('x [mm]')
plt.ylabel('z [mm]')
plt.subplot(122)
plt.hist2d(event[:,1], event[:,2], bins=100, weights=event[:,3], cmap='viridis')
plt.colorbar(label='Energy [GeV]')  # Add a colorbar
plt.scatter(centroids[:, 1], centroids[:, 2], c='red', marker='o', s=50)
plt.xlabel('y [mm]')
plt.ylabel('z [mm]')
plt.title('2D Energy Deposits and Centroids')
plt.savefig('img/2d_energy_deposits.png')

total_energy = assign_deposits_to_centroids(centroids, coordinates, energies)

print(tf.reduce_sum(total_energy))

# 3D Visualization
def plot_3d_voronoi(centroids, deposits, energies):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    mask = energies > 0.1
    scatter = ax.scatter(deposits[:, 0][mask], deposits[:, 1][mask], deposits[:, 2][mask], c=energies[mask], cmap='viridis', s=1, alpha=0.05)
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='red', marker='o', s=5)
    plt.colorbar(scatter, label='Energy [GeV]')
    plt.title('3D Energy Deposits and Centroids')
    plt.savefig('img/3d_voronoi.png')

def voronoi_assignment(centroids, resolution=100):
    grid = generate_grid((resolution, resolution, resolution))  # Shape: (resolution, resolution, 2)
    grid = tf.reshape(grid, (-1, 3))  # Shape: (resolution*resolution, 2)
    grid = tf.cast(grid, tf.float32)
    distances = calculate_distances(centroids, grid)  # Shape: (num_centroids, num_deposits)
    # Find the closest centroid for each deposit
    closest_centroid = tf.argmin(distances, axis=0)  # Shape: (num_deposits,)
    return grid, closest_centroid

grid, closest_centroid = voronoi_assignment(centroids, resolution=100)
assigned_energies = tf.gather(total_energy, closest_centroid)

plot_3d_voronoi(centroids, grid, assigned_energies)

# plot a projection
plt.figure(figsize=(20, 8))
mask = assigned_energies > 0.1
plt.subplot(121)
plt.scatter(grid[:, 0][mask], grid[:, 2][mask], c=assigned_energies[mask], cmap='viridis', s=10, alpha=0.25)
plt.scatter(centroids[:, 0], centroids[:, 2], c='red', marker='o', s=5)
plt.colorbar(label='Energy [GeV]')  # Add a colorbar
plt.xlabel('x [mm]')
plt.ylabel('z [mm]')
plt.title('xz - plane')
plt.subplot(122)
plt.scatter(grid[:, 1][mask], grid[:, 2][mask], c=assigned_energies[mask], cmap='viridis', s=10, alpha=0.25)
plt.scatter(centroids[:, 1], centroids[:, 2], c='red', marker='o', s=5)
plt.colorbar(label='Energy [GeV]')  # Add a colorbar
plt.xlabel('y [mm]')
plt.ylabel('z [mm]')
plt.title('zy - plane')
plt.savefig('img/2d_voronoi.png')

