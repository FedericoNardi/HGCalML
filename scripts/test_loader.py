import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import uproot

# single-file function 
def load_data_unit():
    with uproot.open("/media/disk/g4_showers/unif/photon_showers_96.root:photon_sim") as file:
        evt_id = file['EventID'].array(library='np')
        hit_x = file['x'].array(library='np') # mm
        hit_y = file['y'].array(library='np') # mm
        hit_z = file['z'].array(library='np') # mm
        hit_e = file['dE'].array(library='np')*1e-3 # GeV  
        primary_E = file['primaryE'].array(library='np')*1e-3 # GeV     
    return evt_id, hit_x, hit_y, hit_z, hit_e, primary_E


# Create a dataset from the data, where each entry corresponds to a different event id: new event when ID changes
from tqdm import tqdm

def create_dataset(IDs, x, y, z, dE, prim_E):
    dataset = []
    for i in tqdm(range(len(np.unique(IDs))), desc='Creating dataset: '):
        mask = IDs==i
        dataset.append(np.stack([x[mask], y[mask], z[mask], dE[mask], prim_E[mask]], axis=-1))
    return dataset

def check_energies(dataset):
    print('Dataset length: {}'.format(len(dataset)))
    # Check that the sum of the energy deposits is equal to the primary energy by plotting them
    E_prim = []
    E_nom = []
    for event in tqdm(dataset, desc='Checking energies: '):
        x, y, z, dE, prim_E = zip(*event)
        E_prim.append(np.sum(dE))
        E_nom.append(np.unique(prim_E)[0])
    E_prim = np.array(E_prim)
    E_nom = np.array(E_nom)
    plt.figure(figsize=(12, 6))
    plt.title(f'Out of {len(E_prim)} events')
    plt.subplot(1,2,1)
    plt.hist(E_prim, bins=25, histtype='step', label='Sum of dE')
    plt.hist(E_nom, bins=25, histtype='step', label='Primary E')
    plt.legend()
    plt.xlabel('Energy [GeV]')
    plt.ylabel('Counts')
    plt.subplot(1,2,2)
    print(E_nom.shape)
    print(E_prim.shape)
    plt.hist2d(E_nom, E_prim, bins=25)
    plt.xlabel('Primary Energy [GeV]')
    plt.ylabel('Sum of dE [GeV]')
    plt.savefig('img/energies.png')

# Generate 3D coordinates for each voxel
def generate_grid(grid_size):
    x = -300+600*tf.range(grid_size[0], dtype=tf.float32)/grid_size[0]
    y = -300+600*tf.range(grid_size[1], dtype=tf.float32)/grid_size[1]
    z = tf.range(300, dtype=tf.float32)
    grid_x, grid_y = tf.meshgrid(x, y)
    return tf.stack([grid_x, grid_y], axis=-1)  # Shape: (X, Y, 2)

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

def voronoi_assignment(centroids, resolution=100):
    grid = generate_grid((resolution, resolution))  # Shape: (resolution, resolution, 2)
    grid = tf.reshape(grid, (-1, 2))  # Shape: (resolution*resolution, 2)
    grid = tf.cast(grid, tf.float32)
    distances = calculate_distances(centroids, grid)  # Shape: (num_centroids, num_deposits)
    # Find the closest centroid for each deposit
    closest_centroid = tf.argmin(distances, axis=0)  # Shape: (num_deposits,)
    return grid, closest_centroid

IDs, x, y, z, dE, prim_E = load_data_unit()
dataset = create_dataset(IDs, x, y, z, dE, prim_E)

# Example centroids (random for testing)
num_centroids = 100
centroids = tf.random.uniform((num_centroids, 2), minval=-300, maxval=300, dtype=tf.float32)
event = dataset[50]
coordinates = tf.stack([event[:,0], event[:,2]], axis=-1)
energies = event[:,3]

total_energy = assign_deposits_to_centroids(centroids, coordinates, energies)

# Plot the Voronoi diagram for the centroids, with the color representing the total energy
grid, closest_centroid = voronoi_assignment(centroids, resolution=500)
assigned_energies = tf.gather(total_energy, closest_centroid)
plt.figure(figsize=(18, 8))
plt.suptitle(f'{np.sum(total_energy):.1f} GeV event')
plt.subplot(1,2,1)
plt.hist2d(coordinates[:,0], coordinates[:,1], bins=200, weights=energies, cmap='viridis')
plt.colorbar(label='dE [GeV]')
plt.xlabel('x [mm]')
plt.ylabel('z [mm]')
plt.title('Energy deposits')
plt.subplot(1,2,2)
# fill colors with energy corresponding to assigned centroid
plt.scatter(grid[:,0], grid[:,1], c=assigned_energies, cmap='viridis', s=50)
plt.colorbar(label='dE [GeV]')
plt.scatter(centroids[:,0], centroids[:,1], c='lavender', s=5)
plt.xlabel('x [mm]')
plt.ylabel('z [mm]')
plt.title('Voronoi diagram with centroids and total energy')
plt.savefig(f'img/voronoi_{int(np.sum(total_energy))}.png')
