import tensorflow as tf
import numpy as np
import h5py
from tqdm import tqdm
import os

# tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Data loaders and massaging
def subsample(event, fraction=0.05):
    """ Subsamples a fraction of hits from an event and returns a sparse representation. """
    num_hits = event.shape[0]
    num_samples = int(fraction * num_hits)
    
    if num_samples == 0:
        return np.zeros((1, 4), dtype=np.float32), np.zeros((1,), dtype=np.float32)  # Avoid empty tensors
    
    indices = np.random.choice(num_hits, num_samples, replace=False)
    hits = event[indices]
    
    features = hits[:, [0, 1, 3]] # (X, Z, E0)
    labels = hits[:, 2]  # dE

    return features, labels

def data_generator(folder):
    files = os.listdir(folder)
    # Only containing .h5 files
    files = [f for f in files if f.endswith('.h5')]

    for filename in files:
        with h5py.File(os.path.join(folder, filename), 'r') as f:
            for key in f.keys():
                evt = f[key][()]
                features, labels = evt[:, (0,1,3)], evt[:,2]
                if features.shape[0] == 0:
                    print(f"Empty event in {filename}, key: {key}")
                yield features, labels


def create_tf_dataset(folder, batch_size=8):
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(folder),
        output_signature=(
            tf.TensorSpec(shape=(None, 3), dtype=tf.float32),  # Features (X, Z, E0)
            tf.TensorSpec(shape=(None,), dtype=tf.float32)  # Labels (dE)
        )
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Model definition
# GravNet layer
class GravNetLayer(tf.keras.layers.Layer):
    def __init__(self, feature_dim, k=10):
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
    def __init__(self, feature_dim, k=8):
        self.gravnet = GravNetLayer(feature_dim, k)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.global_exchange = self.global_exchange = tf.keras.layers.GlobalAveragePooling1D()
        self.dense3 = tf.keras.layers.Dense(64, activation='relu')
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

def model():
    inputs = tf.keras.Input(shape=(None, 3))  # Allow variable node sizes
    _x = tf.keras.layers.BatchNormalization()(inputs)
    _x1 = GravNetLayer(32, k=8)(_x)
    _x2 = GravNetLayer(32, k=8)(_x1)
    _x = tf.concat([_x1, _x2], axis=-1)
    _x = tf.keras.layers.GlobalAveragePooling1D()(_x)  # Reduce memory usage
    _x = tf.keras.layers.Dense(128, activation='relu')(_x)
    _x = tf.keras.layers.BatchNormalization()(_x)
    _x = tf.keras.layers.Dense(64, activation='relu')(_x)
    _x = tf.keras.layers.BatchNormalization()(_x)
    _x = tf.keras.layers.Dense(1)(_x)  # Output single value per event

    return tf.keras.Model(inputs, _x)

# ----------------------------
# Execution starts here
# ----------------
model = model()
# Initialize weights to glorot uniform
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mse')

# Train the model
dataset = create_tf_dataset('/media/disk/g4_showers/2D', batch_size=4)
model.fit(dataset, epochs=150)

model.save_weights('gnn_model.h5')