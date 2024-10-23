from genModules import generator, bib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

E0 = 10

# FIXED ENERGY
energy = tf.random.uniform(shape=[], minval=10., maxval=175., dtype=tf.float32)

# 1. Choose parameter space to explore drawing dx, dy, dz from a normal distribution with given mean and standard deviation
means = [10., 10., 50.]
sigmas = [5., 5., 5.]

n_showers = 10

dx = np.abs(np.random.normal(means[0], sigmas[0], n_showers))
dy = np.abs(np.random.normal(means[1], sigmas[1], n_showers))
dz = np.abs(np.random.normal(means[2], sigmas[2], n_showers))

# Initialize bib model once outside the loop for efficiency
bib_model = bib.Model()

# Initialize a list to store features (X, Y, Z, dE) for each shower
features_list = tf.Tensor([],dtype='float32')

# Loop through each shower and generate features
for i in range(n_showers):
    # Generate showers with vectorized input parameters
    gen = generator.Shower(energy, spacing=[dx[i], dy[i], dz[i]], DEBUG=False)
    
    # Get the shower output
    shower = gen()  # Shower shape depends on dx, dy, dz

    X = tf.reshape(gen.X, [-1])
    Y = tf.reshape(gen.Y, [-1])
    Z = tf.reshape(gen.Z, [-1])
    dE = tf.reshape(shower, [-1])

    # Prepare inputs for bib_model in a vectorized way
    input_data = tf.stack([X, Z], axis=-1)
    bib_shower = bib_model.predict(input_data).reshape([-1])

    # Overlay bib_shower onto dE (energy deposition)
    dE = dE + bib_shower

    # Store features (X, Y, Z, dE) for this shower
    features = tf.stack([X, Y, Z, dE], axis=-1)  # Shape (n_points, 4)
    features_list.append(features)

print(f"Feature list shape: {features_list.shape}")

# At this point, `features_list` contains the features for all showers
