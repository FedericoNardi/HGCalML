'''


Compatible with the dataset here:
/eos/home-j/jkiesele/ML4Reco/Gun20Part_NewMerge/train

On flatiron:
/mnt/ceph/users/jkieseler/HGCalML_data/Gun20Part_NewMerge/train

not compatible with datasets before end of Jan 2022

'''

from callback_wrappers import build_callbacks
from experiment_database_manager import ExperimentDatabaseManager
import tensorflow as tf
from argparse import ArgumentParser
# from K import Layer
import numpy as np

from datastructures import TrainData_NanoML

#from tensorflow.keras import Model
from Layers import DictModel

from model_blocks import  pre_selection_model, pre_selection_model2

#K=12 #12 #12

plot_frequency= 100 #600#every 20 minutes approx
record_frequency = 10

nbatch = 100000 #//2
nbatch2 = 100000

def pretrain_model(Inputs,
                   td, 
                   publish,
                   debugplots_after=record_frequency*plot_frequency, 
                   debug_outdir=None):

    publish=None ##FIXME
    
    orig_inputs = td.interpretAllModelInputs(Inputs,returndict=True)
 
    
    
    print('orig_inputs',orig_inputs.keys())
    
    presel = pre_selection_model(orig_inputs,
                             debug_outdir,
                             trainable=True,
                             debugplots_after=debugplots_after,
                             record_metrics=True,
                             #K=K,
                             pf_mode = True,
                             pre_train_mode = False,
                             ext_pf=1 , #large
                             print_time=True,
                             hifp_penalty = 50.,
                             publish = publish
                             )
    
    
    
    print('presel',presel.keys())
    # this will create issues with the output and is only needed if used in a full dim model.
    # so it's ok to pop it here for training
    if 'scatterids' in presel.keys():
        presel.pop('scatterids')
    
    return DictModel(inputs=Inputs, outputs=presel)

import training_base_hgcal
train = training_base_hgcal.HGCalTraining()

publishpath = "jkiesele@lxplus.cern.ch:~/Cernbox/www/files/temp/Oct2022/"
publishpath += [d  for d in train.outputDir.split('/') if len(d)][-1] 

if not train.modelSet():
    train.setModel(pretrain_model,
                   td = train.train_data.dataclass(),
                   publish = publishpath,
                   debug_outdir=train.outputDir+'/intplots')
    
    train.saveCheckPoint("before_training.h5")
    train.setCustomOptimizer(tf.keras.optimizers.Adam())
    #
    train.compileModel(learningrate=1e-4)
    
    train.keras_model.summary()
    
    #start somewhere
    #from model_tools import apply_weights_from_path
    #import os
    #path_to_pretrained = os.getenv("HGCALML")+'/models/pre_selection_jan/KERAS_model.h5'
    #train.keras_model = apply_weights_from_path(path_to_pretrained,train.keras_model)
    



verbosity = 2
import os

samplepath=train.val_data.getSamplePath(train.val_data.samples[0])
# publishpath = 'jkiesele@lxplus.cern.ch:/eos/home-j/jkiesele/www/files/HGCalML_trainings/'+os.path.basename(os.path.normpath(train.outputDir))

from DeepJetCore.training.DeepJet_callbacks import simpleMetricsCallback



cb = [
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/reduction_metrics.html',
        record_frequency = record_frequency ,
        plot_frequency = plot_frequency,
        select_metrics=['*_reduction','*amb_truth_fraction'],#includes time
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/hit_reduction_metrics.html',
        record_frequency = record_frequency ,
        plot_frequency = plot_frequency,
        select_metrics=['*reduction*hits*','*_reduction*lost*'],#includes time
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/noise_metrics.html',
        record_frequency = record_frequency ,
        plot_frequency = plot_frequency,
        select_metrics=['*noise*accuracy','*noise*loss','*noise*reduction'],
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/time.html',
        record_frequency= 10*record_frequency,#doesn't change anyway
        plot_frequency = plot_frequency,
        select_metrics='*time*',
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/losses.html',
        record_frequency = record_frequency ,
        plot_frequency = plot_frequency,
        select_metrics='*_loss',
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    simpleMetricsCallback(
        output_file=train.outputDir+'/val_metrics.html',
        call_on_epoch=True,
        select_metrics='val_*',
        publish=publishpath #no additional directory here (scp cannot create one)
        ),
    
    ]


#cb += build_callbacks(train)

#cb=[]

train.change_learning_rate(1e-3)

train.trainModel(nepochs=2,batchsize=nbatch,additional_callbacks=cb)

train.change_learning_rate(1e-4)

train.trainModel(nepochs=4,batchsize=nbatch,additional_callbacks=cb)

print('reducing learning rate to 1e-5')
train.change_learning_rate(1e-5)


train.trainModel(nepochs=100,batchsize=nbatch2,additional_callbacks=cb)
