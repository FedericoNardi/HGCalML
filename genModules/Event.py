import tensorflow as tf
import tensorflow as tf

from genModules import generator, bib, InflateShower_v2
from genModules.generator import ShowerGNN
from genModules.bib import BibModel
from genModules.Processer_v2 import Voronoi3D

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
        _y = tf.linspace(-200., 200., 50)
        _z = tf.linspace(0., 400., 50)
        _X, _Y, _Z = tf.meshgrid(_x, _y, _z, indexing='ij')
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

        self.inflate_model = InflateShower_v2()
        self.bib_model = BibModel()
        self.voronoi = Voronoi3D()

    def heaviside(self, x, slope=20.0):
        return tf.sigmoid(slope * x)
    
    def __call__(self, centroids: tf.Tensor, E0: tf.Tensor):
        # Generate energy layer input
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
        img /= tf.reduce_sum(img) + 1e-6
        img *= E0

        # Inflate to 3D hits
        hits_3d = self.inflate_model(img)

        # Apply randomized rototranslation
        centered, dE, _ = self.apply_rototranslation(hits_3d)

        # Assign energy to Voronoi cells
        voxels = tf.concat([centered, dE[:, tf.newaxis]], axis=-1)
        energies, volumes = self.voronoi(centroids, voxels, return_volumes=True)

        # Predict BIB
        z = centroids[:, 1]  # vertical
        layer_index = centroids[:, 2]
        query_A = tf.stack([z, layer_index], axis=-1)
        query_B = tf.stack([z, -layer_index], axis=-1)
        bib_A = self.bib_model.predict(query_A)[0]
        bib_B = self.bib_model.predict(query_B)[0]
        bib_density = bib_A + bib_B
        bib_energy = bib_density * volumes[..., tf.newaxis]

        # Total energy
        deposits = energies[..., tf.newaxis] + bib_energy

        return deposits, volumes, energies[..., tf.newaxis]

    def apply_rototranslation(self, hits_3d: tf.Tensor, theta_rad=None, phi_rad=None):
        """
        Rotate the shower around the origin and shift it so that it enters from a random
        point (x0, y0, 0) on the z=0 plane and continues in the (theta, phi) direction.
        """
        # Step 1: Sample random angles
        if theta_rad is None:
            theta_rad = tf.random.uniform([], minval=-np.pi/3, maxval=np.pi/3)
        if phi_rad is None:
            phi_rad = tf.random.uniform([], minval=0, maxval=2*np.pi)

        # Step 2: Sample (x0, y0) entry point at z=0
        x0 = tf.random.uniform([], minval=-200., maxval=200.)
        y0 = tf.random.uniform([], minval=-200., maxval=200.)
        impact_point = tf.stack([x0, y0, 0.])

        # Step 3: Build direction vector
        dir_vec = tf.stack([
            tf.sin(theta_rad) * tf.cos(phi_rad),
            tf.sin(theta_rad) * tf.sin(phi_rad),
            tf.cos(theta_rad)
        ])

        # Step 4: Rodrigues rotation from z-axis to dir_vec
        z_axis = tf.constant([0., 0., 1.], dtype=tf.float32)
        v = tf.linalg.cross(z_axis, dir_vec)
        s = tf.linalg.norm(v)
        c = tf.tensordot(z_axis, dir_vec, axes=1)
        vx = tf.convert_to_tensor([
            [0., -v[2], v[1]],
            [v[2], 0., -v[0]],
            [-v[1], v[0], 0.]
        ])
        R = tf.eye(3) + vx + tf.matmul(vx, vx) * ((1 - c) / (s**2 + 1e-6))

        # Step 5: Rotate voxels and shift so origin lands at (x0, y0, 0)
        rotated = tf.matmul(self.xyz_grid, R)
        shift = impact_point - tf.matmul(tf.constant([[0., 0., 0.]]), R)[0]
        translated = rotated + shift[tf.newaxis, :]

        # Step 6: Mask
        dE = tf.reshape(hits_3d, [-1]) * self.heaviside(translated[:, 2])

        return translated, dE, dir_vec
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

class EventGenerator_v2:
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
        _y = tf.linspace(-200., 200., 50)
        _z = tf.linspace(0., 400., 50)
        _X, _Y, _Z = tf.meshgrid(_x, _y, _z, indexing='ij')
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

        self.inflate_model = InflateShower_v2()
        self.bib_model = BibModel()
        self.voronoi = Voronoi3D()

    def heaviside(self, x, slope=20.0):
        return tf.sigmoid(slope * x)
    
    
    def __call__(self, centroids: tf.Tensor, E0: tf.Tensor):
        # centroids: (B, N, 3), E0: (B,)
        B = tf.shape(centroids)[0]
        N = tf.shape(centroids)[1]

        # Generate batched 2D features for GNN
        energy_layer = tf.repeat(E0[:, tf.newaxis], self.n_points, axis=1)  # (B, P)
        features = tf.tile(self.features, [B, 1, 1])  # (B, P, 3)
        features = tf.concat([
            features[:, :, :2],
            tf.expand_dims(energy_layer, axis=-1)
        ], axis=-1)

        # Predict shower images and normalize
        img = self.shower_model(features, training=False)  # (B, P, 1)
        img = tf.nn.relu(tf.reshape(img, [B, 50, 50]))
        img /= tf.reduce_sum(img, axis=[1, 2], keepdims=True) + 1e-6
        img *= tf.reshape(E0, [-1, 1, 1])  # Scale energy per batch

        # Inflate to 3D hits
        hits_3d = tf.map_fn(self.inflate_model, img)  # (B, M, 3)

        # Apply batched rototranslation
        def transform(hits):
            return self.apply_rototranslation(hits)[:2]  # drop R for now
        transformed = tf.map_fn(transform, hits_3d, fn_output_signature=(tf.float32, tf.float32))
        centered, dE = transformed

        # Combine into voxel format
        voxels = tf.concat([centered, tf.expand_dims(dE, axis=-1)], axis=-1)  # (B, M, 4)
        # Run Voronoi assignment
        def assign_energy(args):
            cent, vox = args
            return self.voronoi(cent, vox)
        fn_output_signature = (
            tf.TensorSpec(shape=(None,), dtype=tf.float32),  # energies shape
            tf.TensorSpec(shape=(None,), dtype=tf.float32),  # volumes shape
        )
        energies, volumes = tf.map_fn(assign_energy, (centroids, voxels), fn_output_signature=fn_output_signature)
        
        # Compute BIB in one call
        z = centroids[:, :, 1]
        layer_index = centroids[:, :, 2]
        query_A = tf.stack([z, layer_index], axis=-1)
        query_B = tf.stack([z, -layer_index], axis=-1)
        combined_query = tf.concat([query_A, query_B], axis=1)  # (B, 2N, 2)
        flat_query = tf.reshape(combined_query, [-1, 2])
        bib_result = self.bib_model.predict(flat_query)[0] # Returns both BIB and std
        bib_result = tf.reshape(bib_result, [B, 2 * N, 1])
        bib_A, bib_B = tf.split(bib_result, 2, axis=1)
        bib_density = bib_A + bib_B

        # Compute BIB energy
        volumes_exp = tf.expand_dims(volumes, axis=-1)
        bib_energy = bib_density * volumes_exp

        # Total deposits
        deposits = tf.expand_dims(energies, axis=-1) + bib_energy  # (B, N, 1)

        return deposits, volumes, tf.expand_dims(energies, axis=-1)

    def apply_rototranslation(self, hits_3d: tf.Tensor, theta_rad=None, phi_rad=None):
        """
        Rotate the shower around the origin and shift it so that it enters from a random
        point (x0, y0, 0) on the z=0 plane and continues in the (theta, phi) direction.
        """
        # Step 1: Sample random angles
        if theta_rad is None:
            theta_rad = tf.random.uniform([], minval=-np.pi/3, maxval=np.pi/3)
        if phi_rad is None:
            phi_rad = tf.random.uniform([], minval=0, maxval=2*np.pi)

        # Step 2: Sample (x0, y0) entry point at z=0
        x0 = tf.random.uniform([], minval=-200., maxval=200.)
        y0 = tf.random.uniform([], minval=-200., maxval=200.)
        impact_point = tf.stack([x0, y0, 0.])

        # Step 3: Build direction vector
        dir_vec = tf.stack([
            tf.sin(theta_rad) * tf.cos(phi_rad),
            tf.sin(theta_rad) * tf.sin(phi_rad),
            tf.cos(theta_rad)
        ])

        # Step 4: Rodrigues rotation from z-axis to dir_vec
        z_axis = tf.constant([0., 0., 1.], dtype=tf.float32)
        v = tf.linalg.cross(z_axis, dir_vec)
        s = tf.linalg.norm(v)
        c = tf.tensordot(z_axis, dir_vec, axes=1)
        vx = tf.convert_to_tensor([
            [0., -v[2], v[1]],
            [v[2], 0., -v[0]],
            [-v[1], v[0], 0.]
        ])
        R = tf.eye(3) + vx + tf.matmul(vx, vx) * ((1 - c) / (s**2 + 1e-6))

        # Step 5: Rotate voxels and shift so origin lands at (x0, y0, 0)
        rotated = tf.matmul(self.xyz_grid, R)
        shift = impact_point - tf.matmul(tf.constant([[0., 0., 0.]]), R)[0]
        translated = rotated + shift[tf.newaxis, :]

        # Step 6: Mask
        dE = tf.reshape(hits_3d, [-1]) * self.heaviside(translated[:, 2])

        return translated, dE, dir_vec
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

