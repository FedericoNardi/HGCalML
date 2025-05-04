from genModules import generator, bib, InflateShower
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from Layers import DictModel
from LossLayers import LLFractionRegressor
from modules.datastructures import TrainData_crilin_reduce

import tensorflow as tf

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

# ------------ MODEL DEFINITION HERE ------------
def interpretAllModelInputs(ilist, returndict=True):
    if not returndict:
        raise ValueError('interpretAllModelInputs: Non-dict output is DEPRECATED. PLEASE REMOVE') 
    '''
    input: the full list of keras inputs
    returns: td
     - rechit feature array
     - t_idx
     - t_energy
     - t_pos
     - t_time
     - t_pid :             non hot-encoded pid
     - t_spectator :       spectator score, higher: further from shower core
     - t_fully_contained : fully contained in calorimeter, no 'scraping'
     - t_rec_energy :      the truth-associated deposited 
                           (and rechit calibrated) energy, including fractional assignments)
     - t_is_unique :       an index that is 1 for exactly one hit per truth shower
     - row_splits
     
    '''
    out = {
        'features':ilist[0],
        't_idx':ilist[2],
        't_energy':ilist[4],
        't_pos':ilist[6],
        't_time':ilist[8],
        't_pid':ilist[10],
        't_spectator':ilist[12],
        't_fully_contained':ilist[14],
        'row_splits':ilist[1]
        }
    print('List kength: ', len(ilist))
    #keep length check for compatibility
    if len(ilist)>16:
        out['t_rec_energy'] = ilist[16]
    if len(ilist)>18:
        out['t_is_unique'] = ilist[18]
        print('UNIQUE IDX: ',out['t_is_unique'])
    if len(ilist) > 20:
        out['t_sig_fraction'] = ilist[20]
    return out

def gravnet_model(Inputs,
                  debug_outdir='test_pipe',
                  plot_debug_every=200
                  # pass_through=True
                  ):
    ####################################################################################
    ##################### Input processing, no need to change much here ################
    ####################################################################################

    pre_selection = interpretAllModelInputs(Inputs,returndict=True)
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

# 1. Choose parameter space to explore drawing dx, dy, dz from a normal distribution with given mean and standard deviation
VERBOSE = False

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os

# Create directory for frames
os.makedirs("frames", exist_ok=True)

# Initialize voxel grid with trainable positions
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

    return X, Y, dX, dY

# Smooth box function
def smooth_box(x, a, b, sharpness=50):
    return tf.sigmoid(sharpness * (x - a)) * (1 - tf.sigmoid(sharpness * (x - b)))

def geometric_loss(X, Y, dX, dY):
    ''' Must be within the 25x25 box, so penalty for each (X,Y) point outside '''
    x_bounds = 1 - smooth_box(X, -13-dX, 13+dX)
    y_bounds = 1 - smooth_box(Y, -13-dY, 13+dY)
    return tf.reduce_sum(x_bounds) + tf.reduce_sum(y_bounds)

def potential_barrier(X, Y, alpha=10, min_dist=1e-3):
    """Repulsive potential term to avoid centroids collapsing onto the same (X, Y) position."""
    X_exp, Y_exp = tf.expand_dims(X, axis=1), tf.expand_dims(Y, axis=1)

    # Compute pairwise Euclidean distance
    dist_sq = (X_exp - tf.transpose(X_exp))**2 + (Y_exp - tf.transpose(Y_exp))**2
    dist_sq += min_dist  # Prevent division by zero

    # Ignore self-interactions
    mask = tf.linalg.band_part(tf.ones_like(dist_sq), 0, -1) - tf.eye(tf.shape(X)[0])
    dist_sq = dist_sq * mask  

    # Repulsive Gaussian-like potential
    barrier = tf.reduce_sum(tf.exp(-alpha * dist_sq))
    return barrier

def rectangle_overlap_loss(X, Y, dX, dY, sharpness=10):
    """Penalizes overlaps between rectangles defined by (X, Y) centroids with sides (dX, dY)."""
    X_exp, Y_exp = tf.expand_dims(X, axis=1), tf.expand_dims(Y, axis=1)
    dX_exp, dY_exp = tf.expand_dims(dX, axis=1), tf.expand_dims(dY, axis=1)

    # Compute pairwise distances
    X_diff = tf.abs(X_exp - tf.transpose(X_exp))
    Y_diff = tf.abs(Y_exp - tf.transpose(Y_exp))

    # Compute average rectangle widths/heights for pairwise comparisons
    dX_pairwise = (dX_exp + tf.transpose(dX_exp)) / 2
    dY_pairwise = (dY_exp + tf.transpose(dY_exp)) / 2

    # Ignore self-overlap
    mask = tf.linalg.band_part(tf.ones_like(X_diff), 0, -1) - tf.eye(tf.shape(X)[0])
    
    # Smooth box penalties in X and Y directions
    loss_X = smooth_box(X_diff * mask, 0, dX_pairwise, sharpness)
    loss_Y = smooth_box(Y_diff * mask, 0, dY_pairwise, sharpness)

    # Total overlap penalty
    return tf.reduce_sum(loss_X + loss_Y)
'''
# Initialize voxel grid
X, Y, dX, dY = initialize_grid()

X = tf.Variable(25*tf.random.uniform(X.shape)-12.5, trainable=True)
Y = tf.Variable(25*tf.random.uniform(Y.shape)-12.5, trainable=True)

# Define optimizer
opt = tf.keras.optimizers.Nadam(learning_rate=0.1)

# Training loop
frames = []
epochs = 500
batches = 64
loss = []
loss_1 = []
loss_2 = []
loss_3 = []

def total_loss(X, Y, dX, dY):
    return geometric_loss(X, Y, dX, dY) + potential_barrier(X,Y) #+ 0.1*rectangle_overlap_loss(X, Y, dX, dY)

for i in range(epochs):
    batch_loss = []
    batch_loss_1 = []
    batch_loss_2 = []
    batch_loss_3 = []
    for batch in tqdm(range(batches), desc=f"Epoch {i+1}/{epochs}"):
        with tf.GradientTape() as tape:
            batch_loss.append(total_loss(X, Y, dX, dY))
        batch_loss_1.append(geometric_loss(X, Y, dX, dY))
        batch_loss_2.append(potential_barrier(X,Y))
        batch_loss_3.append(rectangle_overlap_loss(X, Y, dX, dY))
        gradients = tape.gradient(batch_loss[-1], [X, Y, dX, dY])
        opt.apply_gradients(zip(gradients, [X, Y]))
    loss.append((tf.reduce_mean(batch_loss)))
    loss_1.append((tf.reduce_mean(batch_loss_1)))
    loss_2.append((tf.reduce_mean(batch_loss_2)))
    loss_3.append((tf.reduce_mean(batch_loss_3)))
    # Plot loss
    plt.figure(figsize=(8, 6))
    plt.plot(loss, label="Total Loss")
    plt.plot(loss_1, 'r', label="Geometric Loss")
    plt.plot(loss_2, 'g', label="Potential Barrier")
    plt.plot(loss_3, 'b', label="Rectangle Overlap Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Evolution")
    plt.grid(True)
    plt.savefig(f"loss_3.png")
    plt.close()
    # Save frame every 10 iterations
    if i % 10 == 0:
        plt.figure(figsize=(8, 6))
        plt.xlim(-15, 15)
        plt.ylim(-15, 15)
        plt.scatter(X.numpy(), Y.numpy(), color='blue', alpha=0.6)
        # squares of size dX, dY
        for x, y, dx, dy in zip(X.numpy(), Y.numpy(), dX.numpy(), dY.numpy()):
            plt.gca().add_patch(plt.Rectangle((x-dx/2, y-dy/2), dx, dy, fill='blue', edgecolor=None, lw=0.5, alpha=0.3))
        plt.xlabel("X positions")
        plt.ylabel("Y positions")
        plt.title(f"Centroid Evolution (Iteration {i})")
        plt.grid(True)

        # Save frame as image
        frame_path = f"frames/frame_{i:03d}.png"
        plt.savefig(frame_path)
        plt.close()

        frames.append(frame_path)  # Store frame for GIF

        print(f"(Frame saved)")

# Generate GIF
gif_path = "centroid_movement_3.gif"
with imageio.get_writer(gif_path, mode="I", duration=0.2) as writer:
    for frame in frames:
        writer.append_data(imageio.imread(frame))

print(f"GIF saved as {gif_path}")
'''
# Loop through each shower and generate features
def generate_showers(n_showers, energy, dx, dy, dz):
    row_splits = [0]
    for i in (tqdm(range(n_showers)) if VERBOSE else range(n_showers)):
        # Generate showers with vectorized input parameters
        gen = generator.Shower(energy[i], spacing=[dx[i], dy[i], dz[i]], DEBUG=False)

        # Get the shower output
        shower = gen()  # Shower shape depends on dx, dy, dz

        X = tf.reshape(gen.X, [-1])
        Y = tf.reshape(gen.Y, [-1])
        Z = tf.reshape(gen.Z, [-1])
        true_E = tf.reshape(shower, [-1])

        # Prepare inputs for bib_model in a vectorized way
        input_data = tf.stack([X, Z], axis=-1)
        bib_shower = bib_model.predict(input_data).reshape([-1])

        # Overlay bib_shower onto dE (energy deposition)
        dE = true_E + bib_shower

        # Cut energies below 500keV
        mask = dE > 0.5
        X = X[mask]
        Y = Y[mask]
        Z = Z[mask]
        dE = dE[mask]
        true_E = true_E[mask]
        signal_E = energy[i]*tf.ones_like(X)

        # Store features (X, Y, Z, dE, dE_signal, primary_E) for this shower
        if i == 0:
            features_list = tf.stack([X, Y, Z, dE, true_E, signal_E], axis=-1)

        else:
            features_list = tf.concat([features_list, tf.stack([X, Y, Z, dE, true_E, signal_E], axis=-1)], axis=0)
        row_splits.append(row_splits[-1] + X.shape[0])

    row_splits = tf.constant(row_splits, dtype=tf.int32)
    return features_list, row_splits

def get_feature_list(n_showers=1000, isTraining=True):
    energy = tf.random.uniform([n_showers], minval=0.5, maxval=175)
    means = [10., 10., 50.]
    sigmas = [5., 5., 5.]
    dx = np.abs(np.random.normal(means[0], sigmas[0], n_showers))
    dy = np.abs(np.random.normal(means[1], sigmas[1], n_showers))
    dz = np.abs(np.random.normal(means[2], sigmas[2], n_showers))

    features_list, row_splits = generate_showers(n_showers, energy, dx, dy, dz)

    # At this point, `features_list` contains the features for all showers

    # Prepare inputs for the model
    # feature array 
    zerosf = 0.*features_list[:,0]
    farr =  tf.stack([
        features_list[:,3], # hit_dE
        tf.zeros_like(features_list[:,0]),
        tf.zeros_like(features_list[:,0]), 
        tf.zeros_like(features_list[:,0]),
        tf.zeros_like(features_list[:,0]), 
        features_list[:,0], # hit_x
        features_list[:,1], # hit_y
        features_list[:,2], # hit_z
        tf.zeros_like(features_list[:,0]), 
        tf.zeros_like(features_list[:,0])
    ], axis=1)

    # truth: isSignal, evt_trueE, t_pos (3*zerosf), t_time, t_pid, t_spectator, t_fully_contained (1), evt_dE, is_unique, signalFraction

    if isTraining: # Use numpy instead of tf for training
        inputs = [farr.numpy(), row_splits.numpy(), 
                  np.where(features_list[:,4]>0.5*features_list[:,3], 1, 0), row_splits.numpy(), 
                  features_list[:,4].numpy(), row_splits.numpy(),
                  np.concatenate([features_list[:,0].numpy(), features_list[:,0].numpy(), features_list[:,0].numpy()], axis=-1), row_splits.numpy(),
                  zerosf.numpy(), row_splits.numpy(),
                  zerosf.numpy(), row_splits.numpy(),
                  zerosf.numpy(), row_splits.numpy(),
                  zerosf.numpy()+1., row_splits.numpy(),
                  features_list[:,5] * np.ones_like(features_list[:,0].numpy()), row_splits.numpy(),
                  zerosf.numpy(), row_splits.numpy(),
                  features_list[:,3].numpy()/features_list[:,4].numpy(), row_splits.numpy()]
    else:
        inputs = [farr, row_splits, 
                  tf.where(features_list[:,4]>0.5*features_list[:,3], 1, 0), row_splits, 
                  features_list[:,4], tf.identity(row_splits),
                  tf.concat([features_list[:,0], features_list[:,0], features_list[:,0]], axis=-1), tf.identity(row_splits),
                  tf.zeros_like(features_list[:,0]), tf.identity(row_splits),
                  tf.zeros_like(features_list[:,0]), tf.identity(row_splits),
                  tf.zeros_like(features_list[:,0]), tf.identity(row_splits),
                  tf.zeros_like(features_list[:,0])+1., tf.identity(row_splits),
                  features_list[:,5], tf.identity(row_splits),
                  tf.zeros_like(features_list[:,0]), tf.identity(row_splits),
                  tf.divide(features_list[:,3], features_list[:,4]), tf.identity(row_splits)]
    return inputs

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

# Load weights
keras_model.load_weights("test_pyroot/KERAS_model.h5")

# model not in training mode
keras_model.trainable = False   

# Evaluate the model on a set of generated showers
inputs = get_feature_list(100, isTraining=False)
print(keras_model(inputs))
