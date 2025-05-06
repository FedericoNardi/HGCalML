import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from Layers import DictModel
from LossLayers import LLFractionRegressor
from modules.datastructures import TrainData_crilin_reduce
import imageio
import os

from array import array
from genModules.Event import EventGenerator_v2 as EventGenerator


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

# from K import Layer
import numpy as np

from tensorflow.keras.layers import Dense, Concatenate
# from datastructures import TrainData_crilin_reduce

from DeepJetCore.DJCLayers import StopGradient

from model_blocks import create_outputs, condition_input

from GravNetLayersRagged import RaggedGravNet
from GravNetLayersRagged import DistanceWeightedMessagePassing

from Layers import RaggedGlobalExchange, DistanceWeightedMessagePassing
from Layers import ScaledGooeyBatchNorm2 

from Regularizers import AverageDistanceRegularizer

from model_blocks import extent_coords_if_needed


from DebugLayers import PlotCoordinates

from datastructures import TrainData_crilin_dimension

# ------------ MODEL DEFINITION HERE ------------
def gravnet_model(Inputs,
                  td=TrainData_crilin_dimension(),
                  debug_outdir='test_pipe',
                  plot_debug_every=200
                  # pass_through=True
                  ):
    ####################################################################################
    ##################### Input processing, no need to change much here ################
    ####################################################################################

    pre_selection = td.interpretAllModelInputs(Inputs,returndict=True)
    pre_selection = condition_input(pre_selection, no_scaling=True) 
    pre_selection['features'] = ScaledGooeyBatchNorm2(fluidity_decay=0.1)(pre_selection['features']) #this can decay quickly, the input doesn't change 
    pre_selection['coords'] = ScaledGooeyBatchNorm2(fluidity_decay=0.1)(pre_selection['coords'])

    t_spectator_weight = 0.*pre_selection['t_spectator']
    rs = pre_selection['row_splits']
                               
    x_in = pre_selection['features'] #Concatenate()([pre_selection['coords'],
                           
    x = x_in
    energy = pre_selection['rechit_energy']
    c_coords = pre_selection['coords'] # pre-clustered coordinates
    t_idx = pre_selection['t_idx']
    
    ####################################################################################
    ##################### now the actual model goes below ##############################
    ####################################################################################

    # this is just for reference and occasionally plots the input shower(s)
    c_coords = PlotCoordinates(plot_every = plot_debug_every, outdir = debug_outdir,
                                   name='input_coords')([c_coords,
                                                                    energy,
                                                                    t_idx,
                                                                    rs])

    
    allfeat = []
    
    
    
    #extend coordinates already here if needed
    c_coords = extent_coords_if_needed(c_coords, x, n_cluster_space_coordinates)
        

    for i in range(total_iterations):

        # derive new coordinates for clustering
        x = RaggedGlobalExchange()([x, rs])
        
        x = Dense(64,activation=dense_activation)(x)
        x = Dense(64,activation=dense_activation)(x)
        x = Dense(64,activation=dense_activation)(x)
        x = ScaledGooeyBatchNorm2(**batchnorm_options)(x)
        ### reduction done
        
        n_dims = 6
        #exchange information, create coordinates
        x = Concatenate()([c_coords,x])
        xgn, gncoords, gnnidx, gndist = RaggedGravNet(n_neighbours=n_neighbours[i],
                                                 n_dimensions=n_dims,
                                                 n_filters=64,
                                                 n_propagate=64,
                                                 record_metrics=True,
                                                 coord_initialiser_noise=1e-2,
                                                 use_approximate_knn=False #weird issue with that for now
                                                 )([x, rs])
        
        x = Concatenate()([x,xgn])                                                      
        #just keep them in a reasonable range  
        #safeguard against diappearing gradients on coordinates                                       
        gndist = AverageDistanceRegularizer(strength=1e-4,
                                            record_metrics=False # was true
                                            )(gndist)
                                            
        gncoords = PlotCoordinates(plot_every = plot_debug_every, outdir = debug_outdir,
                                   name='gn_coords_'+str(i))([gncoords, 
                                                                    energy,
                                                                    t_idx,
                                                                    rs]) 
        
        gncoords = StopGradient()(gncoords)
        x = Concatenate()([gncoords,x])           
        
        x = DistanceWeightedMessagePassing([64,64,32,32,16,16],
                                           activation=dense_activation
                                           )([x,gnnidx,gndist])
            
        x = ScaledGooeyBatchNorm2(**batchnorm_options)(x)
        
        x = Dense(64,name='dense_past_mp_'+str(i),activation=dense_activation)(x)
        x = Dense(64,activation=dense_activation)(x)
        x = Dense(64,activation=dense_activation)(x)
        
        x = ScaledGooeyBatchNorm2(**batchnorm_options)(x)
        
        allfeat.append(x)
        
    x = Concatenate()([c_coords]+allfeat)
    #do one more exchange with all
    x = Dense(64,activation=dense_activation)(x)
    x = Dense(64,activation=dense_activation)(x)
    x = Dense(64,activation=dense_activation)(x)
    signal_score = Dense(1, activation='sigmoid', name='signal_score')(x)

    signal_score = LLFractionRegressor(
        name='signal_score_regressor',
        record_metrics=True,
        print_loss=True, 
        mode='regression_mse' # "binary" or "regression_bce" or "regression_mse"
        )([signal_score,pre_selection['t_sig_fraction']])                               
    model_outputs = {'signal_fraction' : signal_score}
    
    return tf.keras.Model(inputs=Inputs, outputs=model_outputs)

# ------------ END OF MODEL DEFINITION ------------

import globals
if False: #for testing
    globals.acc_ops_use_tf_gradients = True 
    globals.knn_ops_use_tf_gradients = True

batchnorm_options={}

#loss options:
loss_options={
    'energy_loss_weight': 0.5,
    'q_min': 1.5,
     #'s_b': 1.2, # Added bkg suppression factor
    'use_average_cc_pos': 0.1,
    'classification_loss_weight':0.,
    'too_much_beta_scale': 1e-5 ,
    'position_loss_weight':1e-5,
    'timing_loss_weight':0.,
    #'beta_loss_scale':.25,
    'beta_push': 0.0 # keep at zero
    }

#elu behaves much better when training
dense_activation='elu'

record_frequency=10
plotfrequency=10 #plots every 1k batches

learningrate = 1e-3
nbatch = 20000
if globals.acc_ops_use_tf_gradients: #for tf gradients the memory is limited
    nbatch = 60000

#iterations of gravnet blocks
n_neighbours=[64,64]
total_iterations = len(n_neighbours)

n_cluster_space_coordinates = 3

# Create directory for frames
os.makedirs("frames", exist_ok=True)



def generate_debug_grid_centroids(batch_size):
    """
    Generate a Muon Collider-style 3D grid of centroids and repeat it across a batch:
    
    Grid definition (in mm):
    - 50 mm spacing in z → 5 layers (0 to 200 mm)
    - 20 mm spacing in x over 800 mm → 21 points (-400 to +400 mm)
    - 20 mm spacing in y over 800 mm → 21 points (-400 to +400 mm)
    
    Args:
    - batch_size (int): the batch dimension B
    
    Returns:
    - Tensor of shape [B, N, 3] with repeated centroid coordinates
    """
    # Z-axis: 5 layers (0 to 200 mm)
    z_vals = tf.linspace(0., 200., 5)
    
    # X and Y: 21 values from -400 to +400 mm (20 mm spacing)
    x_vals = tf.linspace(-400., 400., 21)
    y_vals = tf.linspace(-400., 400., 21)

    # Generate the meshgrid
    gx, gy, gz = tf.meshgrid(x_vals, y_vals, z_vals, indexing='ij')  # [21, 21, 5] each

    # Flatten the grid and stack to shape [N, 3]
    grid = tf.stack([
        tf.reshape(gx, [-1]),
        tf.reshape(gy, [-1]),
        tf.reshape(gz, [-1])
    ], axis=-1)  # [N, 3]

    # Repeat grid for each item in the batch
    grid_batched = tf.tile(tf.expand_dims(grid, axis=0), [batch_size, 1, 1])  # [B, N, 3]

    return grid_batched

event_gen = EventGenerator()

def generate_events(centroids, event_gen=event_gen, DEBUG=True, energy_range=(10., 150.)):
    E0 = tf.random.uniform((centroids.shape[0],), minval=energy_range[0], maxval=energy_range[1])
    deposits, volumes, energy = event_gen(centroids, E0)
    signal_frac = tf.divide(energy, deposits+1e-7)
    
    inputs = tf.concat([
        centroids, 
        volumes[..., tf.newaxis], 
        deposits,
        E0*tf.ones_like(deposits)
        ], axis=-1)
    targets = signal_frac
    return inputs, targets, E0

def get_feature_list_single(inputs, signal_fraction):
    """
    Build model input list for a single event in differentiable form.

    Args:
        inputs: tf.Tensor of shape (N, 6) — [x, y, z, dE, volume, E0]
        signal_fraction: tf.Tensor of shape (N,) — signal weights (0 to 1)

    Returns:
        List[tf.Tensor or tf.RaggedTensor]: input format expected by the model
    """
    x = inputs[:, 0]
    y = inputs[:, 1]
    z = inputs[:, 2]
    volume = inputs[:, 3]
    dE = inputs[:, 4]
    E0 = inputs[0, 5]  # constant per event

    N = tf.shape(inputs)[0]
    row_splits = tf.convert_to_tensor([0, N], dtype=tf.int64)

    # Construct feature tensor matching training format
    features = tf.stack([
        dE,
        tf.zeros_like(dE),
        volume,
        x,
        y,
        z,
        tf.zeros_like(dE),
        tf.zeros_like(dE)
    ], axis=-1)

    # Targets (only used during loss eval)
    t_idx = tf.where(signal_fraction > 0.5, 1.0, 0.0)
    t_energy = tf.fill([N], E0)
    t_pos = tf.zeros((N, 3), dtype=tf.float32)
    t_time = tf.zeros((N, 1), dtype=tf.float32)
    t_pid = tf.zeros((N, 6), dtype=tf.float32); t_pid = tf.tensor_scatter_nd_update(t_pid, [[i,0] for i in range(N)], tf.ones([N]))  # class 0 = photon
    t_spectator = tf.zeros([N, 1], dtype=tf.float32)
    t_fully_contained = tf.ones([N, 1], dtype=tf.float32)
    t_rec_energy = tf.expand_dims(signal_fraction * dE, axis=-1)
    t_is_unique = tf.zeros([N, 1], dtype=tf.float32)

    t_sig_fraction = tf.expand_dims(signal_fraction, axis=-1)

    return [
        features,               row_splits,
        t_idx,                  row_splits,
        t_energy,               row_splits,
        t_pos,                  row_splits,
        t_time,                 row_splits,
        t_pid,                  row_splits,
        t_spectator,            row_splits,
        t_fully_contained,      row_splits,
        t_rec_energy,           row_splits,
        t_is_unique,            row_splits,
        t_sig_fraction,         row_splits
    ]


def get_feature_list_batched(centroids):
    features_list, targets, E0 = generate_events(centroids)  # (B, N, F), (B,)

    batch_inputs = []
    for b in range(features_list.shape[0]):
        inp = get_feature_list_single(features_list[b], targets[b])
        batch_inputs.append(inp)

    return batch_inputs, E0

# Prepare the model
# gradient inputs not required for training

import training_base_hgcal

train = training_base_hgcal.HGCalTraining()

shapes = train.train_data.getNumpyFeatureShapes()
inputdtypes = train.train_data.getNumpyFeatureDTypes()
inputnames = train.train_data.getNumpyFeatureArrayNames()

for i in range(len(inputnames)): #in case they are not named
    if inputnames[i]=="" or inputnames[i]=="_rowsplits":
        inputnames[i]="input_"+str(i)+inputnames[i]

train.keras_inputs=[]
train.keras_inputsshapes=[]

for s,dt,n in zip(shapes,inputdtypes,inputnames):
    train.keras_inputs.append(tf.keras.layers.Input(shape=s, dtype=dt, name=n))
    train.keras_inputsshapes.append(s)

keras_model = gravnet_model(train.keras_inputs)
keras_model.compile(optimizer=tf.keras.optimizers.Adam(lr=learningrate))

# for i, inp in enumerate(keras_model.inputs):
#     print(f"Input {i}: name={inp.name}, shape={inp.shape}, dtype={inp.dtype}")

# Load weights
keras_model.load_weights("test_reco_smear/KERAS_check_model_last.h5")

# model not in training mode
keras_model.trainable = False   

# Evaluate the model on a set of generated showers
batch_size = 2
centroids = generate_debug_grid_centroids(batch_size) 
centroids = tf.Variable(centroids, dtype=tf.float32)


batch_inputs, E0 = get_feature_list_batched(centroids)
for single_input in batch_inputs:
    preds = keras_model(single_input)['signal_fraction']
    deposits = single_input
    dE = deposits[0][:,0]
    sf = preds[:,0]
