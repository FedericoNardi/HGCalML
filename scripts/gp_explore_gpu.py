import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import uproot
from tqdm import tqdm

# Check if the GPU is available
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))

# Define batch and event sizes
num_events = 10000  # Number of events per training batch
num_nodes = 50*50*50  # Number of spatial points per event

def generate_3D_data_from_folder(folder_path, tree_name="photon_sim", bins=(50,50)):
    """
    Extracts all events from all ROOT files in the given folder and returns a list of 3D histograms.
    
    Parameters:
    - folder_path: str, path to the folder containing ROOT files
    - tree_name: str, name of the tree inside the ROOT file
    - bins: tuple, number of bins in (x, y, z) dimensions
    
    Returns:
    - all_events: list of tuples [(X, Y, Z, values, E0) for each event in all files]
    """
    all_events = []
    file_list = glob.glob(os.path.join(folder_path, "*99.root"))

    for file_path in tqdm(file_list):
        with uproot.open(f"{file_path}:{tree_name}") as file:
            x = file["x"].array(library="np")
            y = file["y"].array(library="np")
            z = file["z"].array(library="np")
            dE = file["dE"].array(library="np")
            evt = file["EventID"].array(library="np")
            E0 = file["primaryE"].array(library="np")

        unique_events = np.unique(evt)
        for event_id in unique_events:
            mask = evt == event_id
            H3 = np.histogramdd(
                (x[mask], y[mask], z[mask]), bins=bins, weights=dE[mask]
            )
            edges_x, edges_y, edges_z = H3[1]

            # Compute bin centers
            x_centers = (edges_x[:-1] + edges_x[1:]) / 2
            y_centers = (edges_y[:-1] + edges_y[1:]) / 2
            z_centers = (edges_z[:-1] + edges_z[1:]) / 2

            # Flatten grid and values
            X, Y, Z = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")
            X, Y, Z, values = X.flatten(), Y.flatten(), Z.flatten(), H3[0].flatten()
            evt_e = np.unique(E0[mask])
            all_events.append(np.stack([X, Y, Z, values, evt_e*np.ones_like(X)], axis=-1))

    return np.array(all_events)

# Generate 2D data
def generate_2D_data_from_folder(folder):
    all_events = []
    file_list = glob.glob(os.path.join(folder, "*1*.root"))

    for file_path in tqdm(file_list):
        with uproot.open(f"{file_path}:photon_sim") as file:
            x = file["x"].array(library="np")
            y = file["z"].array(library="np")
            dE = file["dE"].array(library="np")
            evt = file["EventID"].array(library="np")
            E0 = file["primaryE"].array(library="np")

        unique_events = np.unique(evt)
        for event_id in unique_events:
            mask = evt == event_id
            H2 = np.histogram2d(x[mask], y[mask], bins=(50, 50), weights=dE[mask])
            x_edges = H2[1]
            y_edges = H2[2]

            # Compute bin centers
            x_centers = (x_edges[:-1] + x_edges[1:]) / 2
            y_centers = (y_edges[:-1] + y_edges[1:]) / 2

            # Flatten grid and values
            X, Y = np.meshgrid(x_centers, y_centers, indexing="ij")
            X, Y, values = X.flatten(), Y.flatten(), H2[0].flatten()
            evt_e = np.unique(E0[mask])
            all_events.append(np.stack([X, Y, evt_e*np.ones_like(X), values], axis=-1))

    return np.array(all_events)

# Generate 3D data
all_events = generate_2D_data_from_folder("/media/disk/g4_showers/unif")

x = all_events[:, :, 0].astype(np.float32)
y = all_events[:, :, 1].astype(np.float32)
# z = all_events[:, :, 2].astype(np.float32)
E_tiled = all_events[:, :, 2].astype(np.float32)
targets = all_events[:, :, 3].astype(np.float32)
node_features = np.stack([x, y, E_tiled], axis=-1).astype(np.float32)
print('Nodes: ',node_features.shape)
print('X: ',x.shape)
print('Y: ',y.shape)
print('E: ',E_tiled.shape)
print('t: ',targets.shape)

# Define a simple GravNet-like layer
class GravNetLayer(tf.keras.layers.Layer):
    def __init__(self, feature_dim, k=6):
        super().__init__()
        self.feature_dim = feature_dim
        self.k = k
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.feature_dim)
        ])
    
    def call(self, inputs):
        positions, features = inputs[..., :2], inputs[..., 2:]
        
        # Compute distance matrix
        dists = tf.norm(tf.expand_dims(positions, 2) - tf.expand_dims(positions, 1), axis=-1)
        
        # Find k nearest neighbors
        knn_indices = tf.argsort(dists, axis=-1)[..., 1:self.k+1]
        
        # Aggregate features from neighbors
        neighbor_features = tf.gather(features, knn_indices, batch_dims=1)
        aggregated_features = tf.reduce_mean(neighbor_features, axis=-2)
        
        # Transform features
        updated_features = self.mlp(aggregated_features)
        return tf.concat([positions, updated_features], axis=-1)

class GravNetBlock():
    '''
    A block of [GravNet, MessagePassing, BatchNormalization, Dense(128), BatchNormalization, Dense(96), GlobalExchange, Dense(96), BatchNormalization] layers
    '''
    def __init__(self, feature_dim, k=6):
        self.gravnet = GravNetLayer(feature_dim, k)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(96, activation='relu')
        self.global_exchange = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))  # Fix here
        self.dense3 = tf.keras.layers.Dense(96, activation='relu')
        self.bn3 = tf.keras.layers.BatchNormalization()

    def __call__(self, x):
        x = self.gravnet(x)
        x = self.bn1(x)
        x = self.dense1(x)
        x = self.bn2(x)
        x = self.dense2(x)
        x = self.global_exchange(x)
        x = self.dense3(x)
        x = self.bn3(x)
        return x


# Build the GNN Model
inputs_coords = tf.keras.Input(shape=(50*50,3))  # (x, y, E0)
_x = tf.keras.layers.BatchNormalization()(inputs_coords)
x1 = GravNetLayer(64)(_x)
x2 = GravNetLayer(64)(x1)
x3 = GravNetLayer(64)(x2)
x4 = GravNetLayer(64)(x3)
_x = tf.keras.layers.Concatenate()([x1, x2, x3, x4])
_x = tf.keras.layers.Dense(128, activation='relu')(_x)
_x = tf.keras.layers.BatchNormalization()(_x)
_x = tf.keras.layers.Dense(96, activation='relu')(_x)
_x = tf.keras.layers.BatchNormalization()(_x)
_x = tf.keras.layers.Dense(64, activation='relu')(_x)
_x = tf.keras.layers.BatchNormalization()(_x)
_x = tf.keras.layers.Dense(4, activation='relu')(_x)
_x = tf.keras.layers.BatchNormalization()(_x)
_x = tf.keras.layers.Dense(1, activation='relu')(_x)  # Energy deposition
model = tf.keras.Model(inputs_coords, _x)

def dataset_generator():
    for event in all_events:
        yield event[:, :3], event[:, 3]  # Inputs: (x, y, z, E), Targets: energy deposition

batch_size = 16  # Reduce batch size to fit in memory

dataset = tf.data.Dataset.from_generator(dataset_generator, output_types=(tf.float32, tf.float32))
dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

with tf.device('/device:GPU:0'):
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='mse', metrics=['mae'])
    # Train using dataset instead of full array
    model.fit(dataset, epochs=150, validation_data=dataset, callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])

# Save the weights
model.save_weights('gravnet_weights.h5')

'''
foo

# Predictions.  Generate new data for predictions
x = np.random.uniform(-1, 1, (num_events, num_nodes))
y = np.random.uniform(-1, 1, (num_events, num_nodes))
z = np.random.uniform(-1, 1, (num_events, num_nodes))
E = np.random.uniform(1, 10, (num_events, 1))  # One energy value per event

targets = np.exp(-x**2 - y**2 - z**2) * E  # Example deposition function

E_tiled = np.tile(E[:, np.newaxis, :], (1, num_nodes, 1)).astype(np.float32)  # Broadcast E to all nodes

validation_features = np.stack([x, y, z, E_tiled[:,:,0]], axis=-1).astype(np.float32)
predictions = model.predict(validation_features)

# Validation Plots
plt.figure(figsize=(12, 4))

print(targets.shape)
print(predictions.shape)

# True vs. Predicted
plt.subplot(1, 3, 1)
plt.scatter(targets.flatten(), predictions.flatten(), alpha=0.5)
plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
plt.xlabel("True Energy Deposition")
plt.ylabel("Predicted Energy Deposition")
plt.title("True vs. Predicted")

# Residuals Histogram
plt.subplot(1, 3, 2)
residuals = predictions.flatten() - targets.flatten()
plt.hist(residuals, bins=30, alpha=0.7)
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Residuals Histogram")

# Spatial Distribution of Errors
plt.subplot(1, 3, 3)
plt.scatter(x.flatten(), y.flatten(), c=residuals, cmap='coolwarm', alpha=0.5)
plt.colorbar(label="Prediction Error")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Spatial Distribution of Errors")

plt.tight_layout()
plt.savefig('img/metrics.png')'
'''