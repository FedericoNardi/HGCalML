
import tensorflow as tf
import numpy as np
from Layers import DictModel
from LossLayers import LLFractionRegressor
from GravNetLayersRagged import RaggedGravNet, DistanceWeightedMessagePassing
from Layers import ScaledGooeyBatchNorm2
from DeepJetCore.DJCLayers import StopGradient
from model_blocks import condition_input, extent_coords_if_needed
from DebugLayers import PlotCoordinates
from genModules.Event import EventGenerator
from tqdm import tqdm

# ========== Model Parameters ==========
dense_activation = 'elu'
batchnorm_options = {}
n_neighbours = [64, 64]
total_iterations = len(n_neighbours)
n_cluster_space_coordinates = 3

# ========== Inputs Interpretation ==========
def interpretAllModelInputs(ilist, returndict=True):
    out = {
        'features': ilist[0],
        'row_splits': ilist[1],
        't_idx': ilist[2],
        't_energy': ilist[3],
        'coords': ilist[4],
        't_time': ilist[5],
        't_pid': ilist[6],
        't_spectator': ilist[7],
        't_fully_contained': ilist[8],
        'rechit_energy': ilist[9],
        't_is_unique': ilist[10],
        't_sig_fraction': ilist[11],
    }
    return out

# ========== Model Definition ==========
def gravnet_model(Inputs):
    pre = interpretAllModelInputs(Inputs)
    pre = condition_input(pre, no_scaling=True)
    pre['features'] = ScaledGooeyBatchNorm2(fluidity_decay=0.1)(pre['features'])
    pre['coords'] = ScaledGooeyBatchNorm2(fluidity_decay=0.1)(pre['coords'])

    x = pre['features']
    c_coords = extent_coords_if_needed(pre['coords'], x, n_cluster_space_coordinates)
    rs = pre['row_splits']
    energy = pre['rechit_energy']
    t_idx = pre['t_idx']
    
    allfeat = []

    for i in range(total_iterations):
        x = tf.keras.layers.Dense(64, activation=dense_activation)(x)
        x = tf.keras.layers.Dense(64, activation=dense_activation)(x)
        x = ScaledGooeyBatchNorm2(**batchnorm_options)(x)

        x = tf.keras.layers.Concatenate()([c_coords, x])
        xgn, gncoords, gnnidx, gndist = RaggedGravNet(
            n_neighbours=n_neighbours[i],
            n_dimensions=6,
            n_filters=64,
            n_propagate=64,
            coord_initialiser_noise=1e-2
        )([x, rs])

        x = tf.keras.layers.Concatenate()([x, xgn])
        gncoords = StopGradient()(gncoords)
        x = tf.keras.layers.Concatenate()([gncoords, x])

        x = DistanceWeightedMessagePassing([64, 64, 32, 32], activation=dense_activation)(
            [x, gnnidx, gndist]
        )

        x = tf.keras.layers.Dense(64, activation=dense_activation)(x)
        x = ScaledGooeyBatchNorm2(**batchnorm_options)(x)
        allfeat.append(x)

    x = tf.keras.layers.Concatenate()([c_coords] + allfeat)
    x = tf.keras.layers.Dense(64, activation=dense_activation)(x)
    x = tf.keras.layers.Dense(64, activation=dense_activation)(x)

    signal_score = tf.keras.layers.Dense(1, activation='sigmoid', name='signal_score')(x)

    signal_score = LLFractionRegressor(
        name='signal_score_regressor',
        mode='regression_mse'
    )([signal_score, pre['t_sig_fraction']])

    return DictModel(inputs=Inputs, outputs={'signal_fraction': signal_score})

# ========== Generator & Dataset ==========
def generate_showers(centroids, energy, threshold=0.):
    row_splits = [0]
    generator = EventGenerator()
    
    filtered_deposits = []
    filtered_volumes = []
    filtered_signal = []

    for i in tqdm(range(len(energy))):
        dE, vols, sig = generator(centroids, energy[i])  # (n_centroids,)
        mask = dE > threshold

        filtered_deposits = tf.boolean_mask(dE, mask)
        _X = tf.identity(tf.boolean_mask(centroids[:,0][..., tf.newaxis], mask))
        _Y = tf.identity(tf.boolean_mask(centroids[:,1][..., tf.newaxis], mask))
        _Z = tf.identity(tf.boolean_mask(centroids[:,2][..., tf.newaxis], mask))
        filtered_volumes = tf.boolean_mask(vols[..., tf.newaxis], mask)
        filtered_signal = tf.boolean_mask(sig, mask)

        if i==0:
            feature_list = tf.stack([_X, _Y, _Z, filtered_deposits, filtered_signal, energy[i]*tf.ones_like(filtered_signal), filtered_volumes], axis=-1)
        else:
            feature_list = tf.concat(
                [
                    feature_list,
                    tf.stack(
                        [_X, _Y, _Z, filtered_deposits, filtered_signal, energy[i]*tf.ones_like(filtered_signal), filtered_volumes], 
                        axis=-1
                    )
                ], 
                axis=0
            )
        row_splits.append(row_splits[-1]+filtered_deposits.shape[0])
    
    row_splits = tf.constant(row_splits, dtype=tf.int32)

    return feature_list, row_splits

def get_feature_list(centroids, n_showers=10, isTraining=False):
    energy = tf.random.uniform((n_showers, ), minval=10,maxval=150)

    features_list, row_splits = generate_showers(centroids, energy)

    # At this point, `features_list` contains the features for all showers

    # Prepare inputs for the model
    # feature array 
    zerosf = 0.*features_list[:,0]
    farr =  tf.stack([
        features_list[:,3], # hit_dE
        features_list[:,6],             # Voxel volume
        zerosf, 
        zerosf,
        zerosf, 
        features_list[:,0], # hit_x
        features_list[:,1], # hit_y
        features_list[:,2], # hit_z
        zerosf, 
        zerosf
    ], axis=1)

    # truth: isSignal, evt_trueE, t_pos (3*zerosf), t_time, t_pid, t_spectator, t_fully_contained (1), evt_dE, is_unique, signalFraction
    inputs = [farr, tf.identity(row_splits), 
              tf.where(features_list[:,4]>0.5*features_list[:,3], 1, 0), tf.identity(row_splits), 
              features_list[:,4], tf.identity(row_splits),
              tf.concat([
                tf.identity(features_list[:,0]),
                tf.identity(features_list[:,0]),
                tf.identity(features_list[:,0])
              ], axis=-1), tf.identity(row_splits),
              tf.zeros_like(features_list[:,0]), tf.identity(row_splits),
              tf.zeros_like(features_list[:,0]), tf.identity(row_splits),
              tf.zeros_like(features_list[:,0]), tf.identity(row_splits),
              tf.zeros_like(features_list[:,0])+1., tf.identity(row_splits),
              features_list[:,5], tf.identity(row_splits),
              tf.zeros_like(features_list[:,0]), tf.identity(row_splits),
              tf.divide(features_list[:,3], features_list[:,4]+1e-5), tf.identity(row_splits)]
    return inputs

def data_generator(batch_size=1, n_showers=5):
    def generator():
        for _ in range(100000):  # or `while True` if you prefer
            centroids = tf.random.uniform((12, 3), minval=-200, maxval=200)
            features = get_feature_list(centroids, n_showers=n_showers, isTraining=True)
            assert features is not None, "get_feature_list returned None"
            assert len(features) == 22, f"Expected 22 features, got {len(features)}"
            yield tuple(features)

    output_signature = tuple([
        tf.TensorSpec(shape=(None, 10), dtype=tf.float32),  # features
        tf.TensorSpec(shape=(None,), dtype=tf.int32),       # row_splits 1
        tf.TensorSpec(shape=(None,), dtype=tf.int32),       # t_idx
        tf.TensorSpec(shape=(None,), dtype=tf.int32),       # row_splits 2
        tf.TensorSpec(shape=(None,), dtype=tf.float32),     # t_energy
        tf.TensorSpec(shape=(None,), dtype=tf.int32),       # row_splits 3
        tf.TensorSpec(shape=(None,), dtype=tf.float32),     # t_pos x
        tf.TensorSpec(shape=(None,), dtype=tf.float32),     # t_pos y
        tf.TensorSpec(shape=(None,), dtype=tf.float32),     # t_pos z
        tf.TensorSpec(shape=(None,), dtype=tf.int32),       # row_splits 4
        tf.TensorSpec(shape=(None,), dtype=tf.float32),     # t_time
        tf.TensorSpec(shape=(None,), dtype=tf.int32),       # row_splits 5
        tf.TensorSpec(shape=(None,), dtype=tf.float32),     # t_pid
        tf.TensorSpec(shape=(None,), dtype=tf.int32),       # row_splits 6
        tf.TensorSpec(shape=(None,), dtype=tf.float32),     # t_spectator
        tf.TensorSpec(shape=(None,), dtype=tf.int32),       # row_splits 7
        tf.TensorSpec(shape=(None,), dtype=tf.float32),     # t_fully_contained
        tf.TensorSpec(shape=(None,), dtype=tf.int32),       # row_splits 8
        tf.TensorSpec(shape=(None,), dtype=tf.float32),     # t_rec_energy
        tf.TensorSpec(shape=(None,), dtype=tf.int32),       # row_splits 9
        tf.TensorSpec(shape=(None,), dtype=tf.float32),     # is_unique
        tf.TensorSpec(shape=(None,), dtype=tf.int32),       # row_splits 10
        tf.TensorSpec(shape=(None,), dtype=tf.float32),     # signal_fraction
        tf.TensorSpec(shape=(None,), dtype=tf.int32),       # row_splits 11
    ])

    return tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    ).prefetch(tf.data.AUTOTUNE)

# ========== Build and Compile ==========
dataset = data_generator(batch_size=1, n_showers=5)

# Sanity check
example_batch = next(iter(dataset))
print("Got a batch!", type(example_batch))

model = gravnet_model(example_batch)
model.compile(optimizer=tf.keras.optimizers.Nadam(1e-3))
# Train model
model.fit(
    dataset,
    epochs=5,
    steps_per_epoch=1
)

# ========== Train ==========
model.fit(dataset, epochs=10, steps_per_epoch=1)