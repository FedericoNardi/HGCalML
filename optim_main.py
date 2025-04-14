from genModules import generator, bib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from modules.datastructures import TrainData_crilin_reduce
from Layers import LLReduceLoss, LLFractionRegressor
from Layers import DictModel

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
from Layers import LLReduceLoss

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
                  debug_outdir=None,
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
    #c_coords = PlotCoordinates(plot_every = plot_debug_every, outdir = debug_outdir,
    #                               name='input_coords')([c_coords,
    #                                                                energy,
    #                                                                t_idx,
    #                                                                rs])
    
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
                                            
        # gncoords = PlotCoordinates(plot_every = plot_debug_every, outdir = debug_outdir,
        #                            name='gn_coords_'+str(i))([gncoords, 
        #                                                             energy,
        #                                                             t_idx,
        #                                                             rs]) 
        
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
    ''' 
    Working version: signal fraction output
    signal_score = Dense(1, activation='sigmoid', name='signal_score')(x)
    signal_score = LLFractionRegressor(
        name='signal_score_regressor',
        record_metrics=True,
        print_loss=True, 
        mode='regression_mse' # "binary" or "regression_bce" or "regression_mse"
        )([signal_score,pre_selection['t_sig_fraction']])   
                                
    model_outputs = {'signal_fraction' : signal_score}
    
    return DictModel(inputs=Inputs, outputs=model_outputs)
    '''
    return x 

def mlp_reduce(inputs): # AGGIUNGI AD INPUT POSIZIONE DI VOXEL + SIGNAL SCORE
    x = ScaledGooeyBatchNorm2(fluidity_decay=0.1)(inputs)
    x = Dense(1024,activation=dense_activation)(x)
    x = Dense(512,activation=dense_activation)(x)
    x = Dense(128,activation=dense_activation)(x)
    x = Dense(64,activation=dense_activation)(x)
    x = Dense(32,activation=dense_activation)(x)
    x = Dense(16,activation=dense_activation)(x)
    x = Dense(1,activation=tf.keras.activations.relu)(x)
    return x

def gravnet_model_reduced(Inputs,
                  debug_outdir=None,
                  plot_debug_every=200):
    pre_selection = interpretAllModelInputs(Inputs,returndict=True)
    gravnet_output = gravnet_model(Inputs, debug_outdir, plot_debug_every)
    full_output = mlp_reduce(gravnet_output)
    # Add loss layer
    full_output = LLReduceLoss(name='reduce_loss', print_loss=True, record_metrics='True')([full_output, pre_selection['t_energy']])
    model_outputs = {
         'primary_energy': full_output[0]
    }
    return DictModel(inputs=Inputs, outputs=model_outputs)
# ------------ END OF MODEL DEFINITION ------------

# on CPU 
with tf.device('/cpu:0'):

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

    E0 = 10

    # FIXED ENERGY
    energy = tf.random.uniform(shape=[], minval=10., maxval=175., dtype=tf.float32)

    # 1. Choose parameter space to explore drawing dx, dy, dz from a normal distribution with given mean and standard deviation
    means = [10., 10., 50.]
    sigmas = [5., 5., 5.]

    n_showers = 100

    dx = np.abs(np.random.normal(means[0], sigmas[0], n_showers))
    dy = np.abs(np.random.normal(means[1], sigmas[1], n_showers))
    dz = np.abs(np.random.normal(means[2], sigmas[2], n_showers))

    # Initialize bib model once outside the loop for efficiency
    bib_model = bib.Model()

    row_splits = [0]
    # Loop through each shower and generate features
    for i in tqdm(range(n_showers)):
        # Generate showers with vectorized input parameters
        gen = generator.Shower(energy, spacing=[dx[i], dy[i], dz[i]], DEBUG=False)

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

        # Store features (X, Y, Z, dE) for this shower
        if i == 0:
            features_list = tf.stack([X, Y, Z, dE, true_E], axis=-1)

        else:
            features_list = tf.concat([features_list, tf.stack([X, Y, Z, dE, true_E], axis=-1)], axis=0)
        row_splits.append(row_splits[-1] + X.shape[0])

    row_splits = tf.constant(row_splits, dtype=tf.int32)

    # At this point, `features_list` contains the features for all showers

    # Prepare inputs for the model
    # feature array 
    zerosf = 0.*features_list[:,0]
    farr =  tf.stack([
        features_list[:,3], # hit_dE
        zerosf, 
        zerosf, 
        zerosf,
        zerosf, 
        features_list[:,0], # hit_x
        features_list[:,1], # hit_y
        features_list[:,2], # hit_z
        zerosf, 
        zerosf
    ], axis=1)

    # truth: isSignal, evt_trueE, t_pos (3*zerosf), t_time, t_pid, t_spectator, t_fully_contained (1), evt_dE, is_unique

    inputs = [farr, row_splits, 
              tf.where(features_list[:,4]>0.5*features_list[:,3], 1, 0), row_splits, 
              features_list[:,4], row_splits,
              tf.concat(3*[zerosf], axis=-1), row_splits,
              zerosf, row_splits,
              zerosf, row_splits,
              zerosf, row_splits,
              zerosf+1., row_splits,
              E0 + zerosf, row_splits,
              zerosf, row_splits]

    # Prepare the model

#model = gravnet_model_reduced(inputs, debug_outdir=None, plot_debug_every=200)

import training_base_hgcal

with tf.device('/gpu:0'):

    train = training_base_hgcal.HGCalTraining()

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

    keras_model = gravnet_model_reduced(train.keras_inputs)
    train.keras_model = keras_model
    train.keras_model.compile(optimizer=tf.keras.optimizers.Adam(lr=learningrate))
    
    verbosity = 2
    import os

    samplepath=train.val_data.getSamplePath(train.val_data.samples[0])

    publishpath = None
    cb = []

    from callbacks import plotClusterSummary
    cb += [
        plotClusterSummary(
            outputfile=train.outputDir + "/clustering/",
            samplefile=train.val_data.getSamplePath(train.val_data.samples[0]),
            after_n_batches=100
            )
        ]

    cb = []

    train.change_learning_rate(learningrate)

    model, history = train.trainModel(nepochs=80,
                                      batchsize=nbatch,
                                      additional_callbacks=cb)

# Save the model
train.keras_model.save(train.outputDir + '/model.h5')
