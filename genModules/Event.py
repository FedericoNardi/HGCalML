import tensorflow as tf

from genModules import generator, bib, InflateShower
from genModules.generator import ShowerGNN
from genModules.bib import BibModel
from genModules.Processer import Voronoi3D

class EventGenerator:
    def __init__(self):
        # Static grid
        x = tf.linspace(-200., 200., 100)
        z = tf.linspace(0., 400., 100)
        self.shower_x, self.shower_z = tf.meshgrid(x, z, indexing='ij')
        # Prepare input features for the ShowerGNN
        self.n_points = tf.size(self.shower_x)
        self.features = tf.stack([
            tf.reshape(self.shower_x, [-1]),
            tf.reshape(self.shower_z, [-1]),
            tf.fill([self.n_points], 0.)
        ], axis=-1)
        self.features = tf.expand_dims(self.features, axis=0)


        # Precompute full 3D grid (only once)
        _x = tf.linspace(-200., 200., 100)
        _z = tf.linspace(-200., 200., 100)
        _y = tf.linspace(0., 400., 100)
        _X, _Y, _Z = tf.meshgrid(_x, _z, _y, indexing='ij')
        self.xyz_grid = tf.stack([
            tf.reshape(_X, [-1]),
            tf.reshape(_Y, [-1]),
            tf.reshape(_Z, [-1]),
        ], axis=-1)

        # Initialize models once
        shower_model = ShowerGNN()
        shower_model(self.features)
        shower_model.load_weights('genModels/gnn_model_epoch_240.h5')
        self.shower_model = shower_model
        self.shower_model.trainable = False 

        self.inflate_model = InflateShower()
        self.bib_model = BibModel()
        self.voronoi = Voronoi3D()

    def __call__(self, centroids: tf.Tensor, E0: tf.Tensor) -> tf.Tensor:
        # Update features
        energy_layer = tf.fill([self.n_points], E0)
        self.features = tf.stack([
            tf.reshape(self.shower_x, [-1]),
            tf.reshape(self.shower_z, [-1]),
            energy_layer
        ], axis=-1)
        self.features = tf.expand_dims(self.features, axis=0)
        # Predict shower image
        img = self.shower_model(self.features, training=False)
        img = tf.nn.relu(img[0])  # remove negative values, remove batch dim
        img = tf.reshape(img, (100, 100))
        img /= tf.reduce_sum(img) + 1e-6  # avoid division by zero
        img *= E0

        # Inflate to 3D hits
        hits_3d = self.inflate_model(tf.transpose(img))

        # Assign energy to Voronoi cells
        dE = tf.reshape(hits_3d, [-1])
        voxels = tf.concat([self.xyz_grid, dE[:, tf.newaxis]], axis=-1)

        energies, volumes = self.voronoi(centroids, voxels, return_volumes=True)

        # Add BIB
        bib_xy = tf.stack([centroids[:, 1], centroids[:, 2]], axis=-1)
        mirrored_xy = tf.stack([-centroids[:, 1], centroids[:, 2]], axis=-1)

        bib = self.bib_model.predict(bib_xy)[0] + self.bib_model.predict(mirrored_xy)[0]
        bib *= volumes[..., tf.newaxis]
        deposits = energies[..., tf.newaxis] + bib
        return deposits, volumes, energies[..., tf.newaxis]
    
    def __call__new(self, centroids: tf.Tensor, E0: tf.Tensor) -> tf.Tensor:
        """
        Vectorized version of the shower generator.

        Args:
            centroids: (n_centroids, 3) tensor with Voronoi centroids.
            E0: (n_events,) tensor of primary energies.

        Returns:
            deposits: (n_events, n_centroids) tensor of energy deposits.
        """
        print('---> A')
        n_events = tf.shape(E0)[0]
        n_voxels = self.n_points
        print('---> 0')
        # === 1. Create image input features for all events ===
        x_flat = tf.reshape(self.shower_x, [-1])  # shape (n_voxels,)
        z_flat = tf.reshape(self.shower_z, [-1])  # shape (n_voxels,)
        xz = tf.stack([x_flat, z_flat], axis=-1)  # shape (n_voxels, 2)

        # Tile to all events
        xz_tiled = tf.tile(xz[tf.newaxis, :, :], [n_events, 1, 1])  # (n_events, n_voxels, 2)
        E0_tiled = tf.repeat(E0[:, tf.newaxis], repeats=n_voxels, axis=1)  # (n_events, n_voxels)

        features = tf.concat([xz_tiled, E0_tiled[..., tf.newaxis]], axis=-1)  # (n_events, n_voxels, 3)
        print('---> 1')
        # === 2. Predict shower images ===
        img = self.shower_model(features, training=False)  # (n_events, 100*100)
        img = tf.nn.relu(img)  # clamp negatives
        img = tf.reshape(img, [n_events, 100, 100])
        img /= tf.reduce_sum(img, axis=[1, 2], keepdims=True) + 1e-6  # normalize
        img *= E0[:, tf.newaxis, tf.newaxis]  # scale by energy
        print('---> 2')
        # === 3. Inflate to 3D hits (event-wise) ===
        hits_3d = tf.vectorized_map(self.inflate_model, tf.transpose(img, [0, 2, 1]))  # (n_events, ..., ...)

        dE = tf.reshape(hits_3d, [n_events, -1])  # (n_events, n_voxels)
        xyz_grid_tiled = tf.tile(self.xyz_grid[tf.newaxis, :, :], [n_events, 1, 1])  # (n_events, n_voxels, 3)
        voxels = tf.concat([xyz_grid_tiled, dE[..., tf.newaxis]], axis=-1)  # (n_events, n_voxels, 4)
        print('---> 3')
        # === 4. Voronoi integration ===
        energies, volumes = self.voronoi(centroids, voxels, return_volumes=True)  # (n_events, n_centroids)
        print('---> 4')
        # === 5. Add BIB ===
        bib_xy = tf.stack([centroids[:, 1], centroids[:, 2]], axis=-1)  # (n_centroids, 2)
        mirrored_xy = tf.stack([-centroids[:, 1], centroids[:, 2]], axis=-1)

        bib1 = self.bib_model.predict(bib_xy)  # shape (n_centroids,) or (1, n_centroids)
        bib2 = self.bib_model.predict(mirrored_xy)
        bib_sum = tf.convert_to_tensor(bib1) + tf.convert_to_tensor(bib2)  # shape (n_centroids,)

        # Broadcast BIB to (n_events, n_centroids)
        bib = bib_sum[tf.newaxis, :] * volumes  # shape (n_events, n_centroids)

        deposits = energies + bib  # shape (n_events, n_centroids)
        print('---> 5!')
        return deposits