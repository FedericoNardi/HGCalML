
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from Layers import DictModel
from LossLayers import LLFractionRegressor
from modules.datastructures import TrainData_python
from genModules.Event import EventGenerator
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
    print('---> A')
    for i in range(total_iterations):

        # derive new coordinates for clustering
        #x = RaggedGlobalExchange()([x, rs])
        
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
    
    print('---> B')
    x = Concatenate()([c_coords]+allfeat)
    #do one more exchange with all
    x = Dense(64,activation=dense_activation)(x)
    x = Dense(64,activation=dense_activation)(x)
    x = Dense(64,activation=dense_activation)(x)
    signal_score = Dense(1, activation='sigmoid', name='signal_score')(x)
    print('---> C')
    signal_score = LLFractionRegressor(
        name='signal_score_regressor',
        record_metrics=True,
        print_loss=True, 
        mode='regression_mse' # "binary" or "regression_bce" or "regression_mse"
        )([signal_score,pre_selection['t_sig_fraction']])   
    print('---> D')           
    model_outputs = {'signal_fraction' : signal_score}
    print('---> E')
    return DictModel(inputs=Inputs, outputs=model_outputs)

# ------------ END OF MODEL DEFINITION ------------

# Some helper functions now:
# Loop through each shower and generate features
def generate_showers(centroids, energy, threshold=0.):
    row_splits = [0]
    generator = EventGenerator()
    
    filtered_deposits = []
    filtered_volumes = []
    filtered_signal = []

    for i in range(len(energy)):
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

def get_feature_list(centroids, n_showers=100, isTraining=False):
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

# --------- CODE STARTS HERE -----------

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

# Define optimizer
opt = tf.keras.optimizers.Nadam(learning_rate=0.1)


# Prepare the model
import training_base_hgcal
train = training_base_hgcal.HGCalTraining()

N = 5
centroids = tf.random.uniform((N, 3), minval=-200, maxval=200)
centroids = tf.Variable(centroids)

'''
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
'''

input = get_feature_list(centroids, n_showers=10)
keras_model = gravnet_model(input)
keras_model.compile(optimizer=tf.keras.optimizers.Adam(lr=learningrate))

# model in training mode
keras_model.trainable = True   

print('--> Getting Features list!')

# Evaluate the model on a set of generated showers

keras_model.fit()

print('--> Trained model')

keras_model.save('model.keras')

print('--> Saved model')



