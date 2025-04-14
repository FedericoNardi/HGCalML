'''
def plot_voronoi():
    plt.figure(figsize=(10,12))
    plt.subplot(2,1,1)
    plt.hist2d(features[:,1], features[:,2], weights=features[:,3], bins=(100,100), cmap='viridis', norm=mcolors.LogNorm(), range=[(-250,250),(0,500)])
    plt.colorbar()
    plt.subplot(2,1,2)
    plt.hist2d(centroids[:,1], centroids[:,2], weights = energies, cmap='viridis', norm=mcolors.LogNorm(), range=[(-250,250),(0,500)])
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('img/VORONOI_signal.jpg')

# Plot deposits and energies for centroid in different subplots
def plot_img(x, y, c):
    plt.figure(figsize=(10,12))
    plt.subplot(2,1,1)
    plt.hist2d(x, y, weights=c[0][:,0], cmap='viridis')
    plt.colorbar()
    plt.subplot(2,1,2)
    plt.hist2d(x, y, weights=c[1], cmap='viridis')
    plt.colorbar()
    plt.tight_layout
    plt.savefig('img/VORONOI_test.jpg')

# Generate new shower projection 

ShowerModel = ShowerGNN()

# Define input parameters
x = tf.linspace(-200,200,100)
z = tf.linspace(0,400,100)

X, Z = np.meshgrid(x, z)

features = np.c_[X.flatten(), Z.flatten(), E0*np.ones_like(X.flatten())]
# Add batch dimension
features = np.expand_dims(features, axis=0)

ShowerModel(features)
ShowerModel.load_weights('genModels/gnn_model_epoch_240.h5')
img = ShowerModel.predict(features)
img = tf.where(img < 0. , 0., img)
img = tf.reshape(img[0], (100,100))
#normalize
img = img/tf.reduce_sum(img)*E0 #GeV

plt.figure(figsize=(10,10))
plt.imshow(img, cmap='jet', norm=mcolors.LogNorm())
plt.colorbar()
plt.savefig('img/shower_gen.jpg')
plt.close()

model_3d = InflateShower()
hits_3d = model_3d(tf.transpose(img))

bib_model = BibModel()

processer = Voronoi3D()

centroids = tf.random.uniform(shape=(500,3), minval=(-200, -200, 0), maxval=(200,200,400))
centroids = tf.Variable(centroids, dtype=tf.float32)
E0 = (190*np.random.rand())+10

_x = np.linspace(-200, 200, 100)
_z = np.linspace(-200, 200, 100)
_y = np.linspace(0, 400, 100)

_X, _Y, _Z = np.meshgrid(_x, _z, _y)

features = tf.stack([_X.flatten(), _Y.flatten(), _Z.flatten(), tf.reshape(hits_3d, [-1])], axis=-1)

energies, volumes = processer(centroids, features, return_volumes=True) #Volumes should be mm3

deposits = bib_model.predict(tf.stack([centroids[:,1], centroids[:,2]], axis=-1))[0] + bib_model.predict(tf.stack([-centroids[:,1], centroids[:, 2]], axis=-1))[0] 
deposits = deposits * volumes[..., tf.newaxis] # From volumetric density to energy deposit
deposits += energies[..., tf.newaxis]
'''

from genModules.Event import EventGenerator
import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt

# Inputs
points = np.linspace(1,100,10, endpoint=True)
N = 500
B = 5
    
init_start = time.time()
centroids = tf.random.uniform((N, 3), minval=-200, maxval=200)
E0 = tf.random.uniform((B,), minval=5., maxval=150.)  # example energies in GeV
E0 = tf.constant(E0)
generator = EventGenerator()
gen_start = time.time()
with tf.device('/GPU:0'):
    # Generate
    deposits = generator(centroids, E0)

import matplotlib.pyplot as plt
'''
plt.figure(figsize=(10,12))
plt.plot(points, gen_times, label='generation time')
plt.plot(points, init_times, label='initialization time')
plt.legend()
plt.savefig('img/diagnostic_time_n_showers')
'''







