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

from helpers import generate_debug_grid_centroids, generate_events, get_feature_list_single

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

event_gen = EventGenerator()

# Prepare the model

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
centroids = tf.random.uniform((1, 2205, 3), minval=-400, maxval=400) # generate_debug_grid_centroids() 
# Shift centroids z to 0
centroids = tf.concat([tf.random.uniform((1, 2205, 2), minval=-400, maxval=400), tf.random.uniform((1, 2205, 1), minval=0, maxval=200)], axis=-1)
print(centroids.shape)
centroids = tf.Variable(centroids, dtype=tf.float32)

# Define custom loss
def loss_mse(E_pred, E_target):
    return tf.reduce_mean(tf.abs(E_target - E_pred)/E_target)

def full_loss(centroids, model=keras_model):
    inputs, targets, E0 = generate_events(centroids, event_gen=event_gen, DEBUG=False)
    input_list = get_feature_list_single(inputs[0], targets[0])
    # Get model predictions
    outputs = keras_model(input_list)
    E_pred = tf.reduce_sum(outputs['signal_fraction'][:,0]*input_list[0][:,0]) # E_pred = signal_fraction * dE
    print('--> ',E_pred)
    print('--> ',E0)
    # Compute std of volumes
    vols = input_list[0][:,2]
    vol_loss = tf.math.reduce_std(vols)
    # Compute loss
    loss = loss_mse(E_pred, E0) + 1e-2*vol_loss
    return loss

lr = 1e-6

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
n_epochs = 500
burn_in = 10
accum_steps = 64

# @tf.function
def train_step(centroids):
    grad_accum = [tf.zeros_like(c) for c in [centroids]]
    total_loss = 0.0

    for _ in range(accum_steps):
        with tf.GradientTape() as tape:
            loss = full_loss(centroids) / accum_steps
        grads = tape.gradient(loss, [centroids])
        grad_accum = [ga + g for ga, g in zip(grad_accum, grads)]
        total_loss += loss

    optimizer.apply_gradients(zip(grad_accum, [centroids]))
    return total_loss

total_loss = []
for epoch in range(n_epochs):
    loss = train_step(centroids)
    total_loss.append(loss.numpy())
    
    if epoch % 10 == 0:
        # print(f"Epoch {epoch:3d} | Loss = {loss.numpy():.4f}")
        # Plot loss curve
        plt.figure()
        plt.plot(total_loss)
        plt.savefig(f'img/debug/pipe_loss_{lr}.jpg')
        plt.close()
        # Plot centroids 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(centroids[:, :, 0], centroids[:, :, 1], centroids[:, :, 2], c='r', marker='o')
        ax.set_xlabel('X') 
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Centroids')
        plt.savefig(f'frames/pipe_centroids_{epoch}.jpg')

# Save final centroids as csv
centroids_np = centroids.numpy()
centroids_np = centroids_np.reshape(-1, 3)
np.savetxt('centroids.csv', centroids_np, delimiter=',', header='x,y,z', comments='')
