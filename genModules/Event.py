import tensorflow as tf

from genModules import generator, bib, InflateShower
from genModules.generator import ShowerGNN
from genModules.bib import BibModel
from genModules.Processer import Voronoi3D

import os
import matplotlib.pyplot as plt
import numpy as np

class EventGenerator:
    def __init__(self):
        # Shower image prediction grid (depth x vertical)
        x = tf.linspace(-200., 200., 50)  # shower_x: depth (maps to centroid[:,2])
        z = tf.linspace(0., 400., 50)     # shower_z: vertical (maps to centroid[:,1])
        self.shower_x, self.shower_z = tf.meshgrid(x, z, indexing='ij')
        self.n_points = tf.size(self.shower_x)
        self.features = tf.stack([
            tf.reshape(self.shower_x, [-1]),
            tf.reshape(self.shower_z, [-1]),
            tf.fill([self.n_points], 0.)
        ], axis=-1)
        self.features = tf.expand_dims(self.features, axis=0)

        # Grid for inflation (x=horizontal, y=vertical, z=depth)
        _x = tf.linspace(-200., 200., 50)
        _z = tf.linspace(-200., 200., 50)
        _y = tf.linspace(0., 400., 50)
        _X, _Y, _Z = tf.meshgrid(_x, _z, _y, indexing='ij')
        self.xyz_grid = tf.stack([
            tf.reshape(_X, [-1]),
            tf.reshape(_Y, [-1]),
            tf.reshape(_Z, [-1]),
        ], axis=-1)

        # Load models
        shower_model = ShowerGNN()
        shower_model(self.features)  # build once
        shower_model.load_weights('genModels/gnn_model_epoch_240.h5')
        self.shower_model = shower_model
        self.shower_model.trainable = False

        self.inflate_model = InflateShower()
        self.bib_model = BibModel()
        self.voronoi = Voronoi3D()

    def __call__(self, centroids: tf.Tensor, E0: tf.Tensor):
        energy_layer = tf.fill([self.n_points], E0)
        self.features = tf.stack([
            tf.reshape(self.shower_x, [-1]),
            tf.reshape(self.shower_z, [-1]),
            energy_layer
        ], axis=-1)
        self.features = tf.expand_dims(self.features, axis=0)

        # Predict shower image
        img = self.shower_model(self.features, training=False)
        img = tf.nn.relu(img[0])
        img = tf.reshape(img, (50, 50))
        img /= tf.reduce_sum(img)
        img *= E0

        # Inflate to 3D hits
        hits_3d = self.inflate_model(img)
        dE = tf.reshape(hits_3d, [-1])
        voxels = tf.concat([self.xyz_grid, dE[:, tf.newaxis]], axis=-1)

        # Assign signal energy
        energies, volumes = self.voronoi(centroids, voxels, return_volumes=True)

        # Predict BIB
        z = centroids[:, 1]  # physical z
        layer_index = tf.cast(centroids[:, 2], tf.float32)
        query_A = tf.stack([z, layer_index], axis=-1)
        query_B = tf.stack([z, -layer_index], axis=-1)
        bib_A = self.bib_model.predict(query_A)[0]
        bib_B = self.bib_model.predict(query_B)[0]
        bib_density = bib_A + bib_B
        bib_energy = bib_density * volumes[..., tf.newaxis]

        deposits = energies[..., tf.newaxis] + bib_energy # All in GeV

        return deposits, volumes, energies[..., tf.newaxis]
    
def plot_voronoi_deposits(centroids, dE, event_id=0, title="Voronoi Deposits", save_prefix="img/debug/deposits"):
    """
    Plot 3D scatter and 2D projections (X-Z, Y-Z, X-Y) of energy deposits after Voronoi assignment.

    Parameters:
    - centroids: np.ndarray of shape (N, 3)
    - dE: np.ndarray of shape (N,)
    - event_id: int
    - title: plot title prefix
    - save_prefix: path prefix to save images
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    os.makedirs("img/debug", exist_ok=True)

    x, y, z = centroids[:, 0], centroids[:, 1], centroids[:, 2]
    log_dE = np.log10(dE + 1e-6)

    fig = plt.figure(figsize=(24, 6))

    # 1. 3D scatter
    ax = fig.add_subplot(141, projection='3d')
    sc = ax.scatter(x, y, z, c=log_dE, cmap='plasma', s=10, alpha=0.8)
    fig.colorbar(sc, ax=ax, label='log10(E deposit)')
    ax.set_title(f"{title} 3D")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # 2. X-Z projection
    ax2 = fig.add_subplot(142)
    h_xz = ax2.hist2d(x, z, bins=50, weights=dE, cmap='plasma', norm=plt.matplotlib.colors.LogNorm())
    fig.colorbar(h_xz[3], ax=ax2, label='E deposit [X-Z]')
    ax2.set_title(f"{title} X-Z")
    ax2.set_xlabel("x")
    ax2.set_ylabel("z")

    # 3. Y-Z projection
    ax3 = fig.add_subplot(143)
    h_yz = ax3.hist2d(y, z, bins=50, weights=dE, cmap='plasma', norm=plt.matplotlib.colors.LogNorm())
    fig.colorbar(h_yz[3], ax=ax3, label='E deposit [Y-Z]')
    ax3.set_title(f"{title} Y-Z")
    ax3.set_xlabel("y")
    ax3.set_ylabel("z")

    # 4. X-Y projection
    ax4 = fig.add_subplot(144)
    h_xy = ax4.hist2d(x, y, bins=50, weights=dE, cmap='plasma', norm=plt.matplotlib.colors.LogNorm())
    fig.colorbar(h_xy[3], ax=ax4, label='E deposit [X-Y]')
    ax4.set_title(f"{title} X-Y")
    ax4.set_xlabel("x")
    ax4.set_ylabel("y")

    fname = f"{save_prefix}_{event_id:03d}.jpg"
    plt.tight_layout()
    plt.savefig(fname, dpi=250)
    plt.close()


