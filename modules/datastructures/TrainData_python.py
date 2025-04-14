from DeepJetCore.TrainData import TrainData, fileTimeOut
from DeepJetCore import SimpleArray
import numpy as np
import uproot3 as uproot

import os
import pickle
import gzip
import pandas as pd

from datastructures import TrainData_NanoML

# Helper generator function
def generate_showers(centroids, energy, threshold=0.1):
    row_splits = [0]
    generator = EventGenerator()
    
    filtered_deposits = []
    filtered_volumes = []
    filtered_signal = []

    for i in range(len(energy)):
        dE, vols, sig = generator(centroids, energy[i])  # (n_centroids,)
        mask = dE > threshold

        filtered_deposits = tf.boolean_mask(dE, mask)
        _X = tf.boolean_mask(centroids[:,0][..., tf.newaxis], mask)
        _Y = tf.boolean_mask(centroids[:,1][..., tf.newaxis], mask)
        _Z = tf.boolean_mask(centroids[:,2][..., tf.newaxis], mask)
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

class TrainData_python(TrainData_NanoML):
    def __init__(self, centroids):
        TrainData_NanoML.__init__(self)
        self.centroids = centroids

    def convertFromSourceFile(self, filename, weighterobjects, istraining, treename="converted_photons"):
        
        '''
        
        hit_x, hit_y, hit_z: the spatial coordinates of the voxel centroids that registered the hit
        hit_dE: the energy registered in the voxel (signal + BIB noise)
        recHit_dE: the 'reconstructed' hit energy, i.e. the energy deposited by signal only
        evt_dE: the total energy deposited by the signal photon in the calorimeter
        evt_ID: an int label for each event -only for bookkeeping, should not be needed
        isSignal: a flag, -1 if only BIB noise, 0 if there is also signal hit deposition

        '''
        return get_feature_list(self.centroids)
        

        
        
        
        
        
    
    
    
 
