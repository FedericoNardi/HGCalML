# modules/datastructures/TrainData_PhotonSim.py
from DeepJetCore.TrainData import TrainData
from genModules.Event import EventGenerator
from tqdm import tqdm
import tensorflow as tf
import numpy as np

# helper functions
# Loop through each shower and generate features
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

class TrainData_Python(TrainData):
    def __init__(self, centroids):
        super().__init__()
        self.description = "PhotonSim reconstructed showers"
        self.truthclasses = ['signalFraction']
        # self.registerBranches(['isSignal'])  # You can add more here
        self.weightbranchX = 'none'  # if you're not weighting samples
        self.centroids = centroids

    def convertFromSourceFile(self, filename, weighterobjects):
        """
        Called by DeepJetCore to parse and load your input data.
        Replace this with your simulation generator.
        """

        inputs = get_feature_list(self.centroids)

        # Assign
        self.x = [inputs[0]]  # feature array
        self.rs = [inputs[1]]  # row splits

        self.y = [inputs[18]]  # isSignal
        self.truthdata = [inputs[4], inputs[6], inputs[8]]  # pick only relevant ones

        # you can add weights if needed:
        self.w = [np.ones_like(inputs[18])]  # dummy weights

        self.originaltruth = self.y[0]  # Needed for loss printing etc.

        # Cast to np arrays, if not already
        self._finalize()

    def _finalize(self):
        self.x = [x.numpy() if hasattr(x, 'numpy') else x for x in self.x]
        self.rs = [r.numpy() if hasattr(r, 'numpy') else r for r in self.rs]
        self.y = [y.numpy() if hasattr(y, 'numpy') else y for y in self.y]
        self.truthdata = [td.numpy() if hasattr(td, 'numpy') else td for td in self.truthdata]
        self.w = [w.numpy() if hasattr(w, 'numpy') else w for w in self.w]
