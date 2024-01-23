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
from datastructures import TrainData_crilin

from callbacks import plotClusterSummary
from DeepJetCore.DJCLayers import StopGradient

from model_blocks import create_outputs, condition_input

from GravNetLayersRagged import RaggedGravNet
from GravNetLayersRagged import DistanceWeightedMessagePassing

from Layers import RaggedGlobalExchange, DistanceWeightedMessagePassing
from Layers import ScaledGooeyBatchNorm2 #make a new line
from Regularizers import AverageDistanceRegularizer

from model_blocks import extent_coords_if_needed, re_integrate_to_full_hits


from LossLayers import LLFullObjectCondensation

from DebugLayers import PlotCoordinates

from DeepJetCore.wandb_interface import wandb_wrapper as wandb
wandb.active=False

import globals
if False: #for testing
    globals.acc_ops_use_tf_gradients = True 
    globals.knn_ops_use_tf_gradients = True


batchnorm_options={}

#loss options:
loss_options={
    'energy_loss_weight': 0.,
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

learningrate = 1e-2
nbatch = 50000
if globals.acc_ops_use_tf_gradients: #for tf gradients the memory is limited
    nbatch = 60000

#iterations of gravnet blocks
n_neighbours=[64,64]
total_iterations = len(n_neighbours)

n_cluster_space_coordinates = 3


def gravnet_model(Inputs,
                  td,
                  debug_outdir=None,
                  plot_debug_every=200,
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

    print('x_in', x_in.shape)
                           
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
                                            record_metrics=True
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
    
    
    #######################################################################
    ########### the part below should remain almost unchanged #############
    ########### of course with the exception of the OC loss   #############
    ########### weights                                       #############
    #######################################################################
    
    #use a standard batch norm at the last stage
    x = ScaledGooeyBatchNorm2(**batchnorm_options)(x)
    #x = Concatenate()([c_coords]+[x])
    
    pred_beta, pred_ccoords, pred_dist,\
    pred_energy_corr, pred_energy_low_quantile, pred_energy_high_quantile,\
    pred_pos, pred_time, pred_time_unc, pred_id = create_outputs(x, n_ccoords=n_cluster_space_coordinates, fix_distance_scale=True)
    
    print("======= outputs created =======")
    print("---> pred_beta: ", pred_beta)
    print("---> pred_ccoords: ", pred_ccoords)
    print("---> pred_dist: ", pred_dist)
    print("---> pred_energy_corr: ", pred_energy_corr)
    print("---> pred_energy_low_quantile: ", pred_energy_low_quantile)
    print("---> pred_energy_high_quantile: ", pred_energy_high_quantile)
    print("---> pred_pos: ", pred_pos)
    print("---> pred_time: ", pred_time)
    print("---> pred_time_unc: ", pred_time_unc)
    print("---> pred_id: ", pred_id)
    print("=====================")


    # loss
    pred_beta = LLFullObjectCondensation(scale=1.,
                                         use_energy_weights=True,
                                         record_metrics=True,
                                         print_loss=True,
                                         name="FullOCLoss",
                                         **loss_options
                                         )(  # oc output and payload
        [pred_beta, pred_ccoords, pred_dist,
         pred_energy_corr,pred_energy_low_quantile,pred_energy_high_quantile,
         pred_pos, pred_time, pred_time_unc,
         pred_id] +
        [energy]+
        # truth information
        [pre_selection['t_idx'] ,
         pre_selection['t_energy'] ,
         pre_selection['t_pos'] ,
         pre_selection['t_time'] ,
         pre_selection['t_pid'] ,
         pre_selection['t_spectator'],
         pre_selection['t_fully_contained'],
         pre_selection['t_rec_energy'],
         pre_selection['t_is_unique'],
         pre_selection['row_splits']])
                                         
    #fast feedback
    pred_ccoords = PlotCoordinates(plot_every=plot_debug_every, outdir = debug_outdir,
                    name='condensation')([pred_ccoords, pred_beta,pre_selection['t_idx'],
                                          rs])                                    

    model_outputs = re_integrate_to_full_hits(
        pre_selection,
        pred_ccoords,
        pred_beta,
        pred_energy_corr,
        pred_energy_low_quantile,
        pred_energy_high_quantile,
        pred_pos,
        pred_time,
        pred_id,
        pred_dist,
        dict_output=True,
        #is_preselected_dataset=is_preselected
        )
    
    return tf.keras.Model(inputs=Inputs, outputs=model_outputs)
    
import training_base_hgcal
train = training_base_hgcal.HGCalTraining()

if not train.modelSet():
    train.setModel(gravnet_model,
                   td=train.train_data.dataclass(),
                   debug_outdir=train.outputDir+'/intplots')
    
    train.setCustomOptimizer(tf.keras.optimizers.Nadam(clipnorm=1.,epsilon=1e-2))
    #
    train.compileModel(learningrate=1e-4)
    
    train.keras_model.summary()
    
verbosity = 2
import os

samplepath=train.val_data.getSamplePath(train.val_data.samples[0])
# publishpath = 'jkiesele@lxplus.cern.ch:/eos/home-j/jkiesele/www/files/HGCalML_trainings/'+os.path.basename(os.path.normpath(train.outputDir))


# publishpath = "jkiesele@lxplus.cern.ch:~/Cernbox/www/files/temp/July2022_jk/"
#publishpath = "fnardi@lxplus.cern.ch:/eos/user/f/fnardi/www/"

#publishpath += [d  for d in train.outputDir.split('/') if len(d)][-1] 
publishpath = None
cb = []


#cb += [plotClusteringDuringTraining(
#    use_backgather_idx=8 + i,
#    outputfile=train.outputDir + "/localclust/cluster_" + str(i) + '_',
#    samplefile=samplepath,
#    after_n_batches=500,
#    on_epoch_end=False,
#    publish=None,
#    use_event=0)
#    for i in [0, 2, 4]]
#
#cb += [
#    plotEventDuringTraining(
#        outputfile=train.outputDir + "/condensation/c_"+str(i),
#        samplefile=samplepath,
#        after_n_batches=2*plotfrequency,
#        batchsize=200000,
#        on_epoch_end=False,
#        publish=None,
#        use_event=i)
#for i in range(5)
#]
#

from callbacks import plotClusterSummary

cb += [
    plotClusterSummary(
        outputfile=train.outputDir + "/clustering/",
        samplefile=train.val_data.getSamplePath(train.val_data.samples[0]),
        after_n_batches=100
        )
    ]

train.change_learning_rate(learningrate)

model, history = train.trainModel(nepochs=50,
                                  batchsize=nbatch,
                                  add_progbar=True,
                                  additional_callbacks=cb)



# Note the submodel here its not just train.keras_model
for l in train.keras_model.layers:
    if 'FullOCLoss' in l.name:
        l.q_min/=2.

train.change_learning_rate(1e-3)
#nbatch = 160000
if globals.acc_ops_use_tf_gradients: #for tf gradients the memory is limited
    nbatch = 60000

model, history = train.trainModel(nepochs=121,
                                  batchsize=nbatch,
                                  additional_callbacks=cb)
    
