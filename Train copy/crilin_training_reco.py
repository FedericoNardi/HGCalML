'''
compatible with the dataset here:
/eos/home-j/jkiesele/ML4Reco/Gun20Part_NewMerge/train

On flatiron:
/mnt/ceph/users/jkieseler/HGCalML_data/Gun20Part_NewMerge/train

not compatible with datasets before end of Jan 2022

'''

import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

# from K import Layer
import numpy as np

from tensorflow.keras.layers import Dense, Concatenate
from datastructures import TrainData_crilin_reduce

from callbacks import plotClusterSummary
from DeepJetCore.DJCLayers import StopGradient

from model_blocks import create_outputs, condition_input

from GravNetLayersRagged import RaggedGravNet
from GravNetLayersRagged import DistanceWeightedMessagePassing

from Layers import RaggedGlobalExchange, DistanceWeightedMessagePassing
from Layers import ScaledGooeyBatchNorm2 
from Layers import LLReduceLoss

from Regularizers import AverageDistanceRegularizer

from model_blocks import extent_coords_if_needed, re_integrate_to_full_hits


from LossLayers import LLFullObjectCondensation

from DebugLayers import PlotCoordinates

from DeepJetCore.wandb_interface import wandb_wrapper as wandb


# ------------ MODEL DEFINITION HERE ------------ 

def gravnet_model(Inputs,
                  td,
                  debug_outdir=None,
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

    # signal_score = LLFractionRegressor(
    #     name='signal_score_regressor',
    #     record_metrics=True,
    #     print_loss=True, 
    #     mode='regression_mse' # "binary" or "regression_bce" or "regression_mse"
    #     )([signal_score,pre_selection['t_sig_fraction']])                               
    # model_outputs = {'signal_fraction' : signal_score}
    
    #return tf.keras.Model(inputs=Inputs, outputs=model_outputs)
    signal_score = Dense(1, activation='sigmoid', name='signal_score')(x)
    return signal_score

def mlp_reduce(inputs): # AGGIUNGI AD INPUT POSIZIONE DI VOXEL + SIGNAL SCORE
    x = ScaledGooeyBatchNorm2(fluidity_decay=0.1)(inputs)
    x = Dense(1024,activation=dense_activation)(x)
    x = Dense(512,activation=dense_activation)(x)
    x = Dense(128,activation=dense_activation)(x)
    x = Dense(64,activation=dense_activation)(x)
    x = Dense(32,activation=dense_activation)(x)
    x = Dense(16,activation=dense_activation)(x)
    x = Dense(1,activation='ReLU')(x)
    return x

def gravnet_model_reduced(Inputs,
                  td,
                  debug_outdir=None,
                  plot_debug_every=200):
    pre_selection = td.interpretAllModelInputs(Inputs,returndict=True)
    gravnet_output = gravnet_model(Inputs, td, debug_outdir, plot_debug_every)
    full_output = mlp_reduce(gravnet_output)
    # Add loss layer
    full_output = LLReduceLoss(name='reduce_loss', print_loss=True, record_metrics='True')([full_output, pre_selection['t_energy']])
    model_outputs = {
         'primary_energy': full_output[0]
    }
    return tf.keras.Model(inputs=Inputs, outputs=model_outputs)

# ------------ MODEL DEFINITION END ------------

wandb.init( 
    project='HGCal_light'
 )

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
nbatch = 50000
if globals.acc_ops_use_tf_gradients: #for tf gradients the memory is limited
    nbatch = 60000

#iterations of gravnet blocks
n_neighbours=[64,64]
total_iterations = len(n_neighbours)

n_cluster_space_coordinates = 3
    
import training_base_hgcal

train = training_base_hgcal.HGCalTraining()

if not train.modelSet():
    train.setModel(gravnet_model_reduced,
                   td=train.train_data.dataclass(),
                   debug_outdir=train.outputDir+'/intplots')
    print("---> TRAIN DATA: ",train.train_data.dataclass())
    
    train.setCustomOptimizer(tf.keras.optimizers.Nadam(clipnorm=1.,epsilon=1e-2))
    #
    train.compileModel(learningrate=1e-3)
    
    train.keras_model.summary()

# Plot model graph
#from tensorflow.keras.utils import plot_model
#plot_model(train.keras_model, to_file=train.outputDir+'/model_plot.png', show_shapes=True, show_layer_names=True)

    
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

model, history = train.trainModel(nepochs=250,
                                  batchsize=nbatch,
                                  add_progbar=True,
                                  additional_callbacks=cb)