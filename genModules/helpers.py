import tensorflow as tf

def generate_debug_grid_centroids(batch_size=1):
    """
    Generate a Muon Collider-style 3D grid of centroids and repeat it across a batch:
    
    Grid definition (in mm):
    - 50 mm spacing in z → 5 layers (0 to 200 mm)
    - 20 mm spacing in x over 800 mm → 21 points (-400 to +400 mm)
    - 20 mm spacing in y over 800 mm → 21 points (-400 to +400 mm)
    
    Args:
    - batch_size (int): the batch dimension B
    
    Returns:
    - Tensor of shape [B, N, 3] with repeated centroid coordinates
    """
    # Z-axis: 5 layers (0 to 200 mm)
    z_vals = tf.linspace(0., 200., 5)
    
    # X and Y: 21 values from -400 to +400 mm (20 mm spacing)
    x_vals = tf.linspace(-400., 400., 21)
    y_vals = tf.linspace(-400., 400., 21)

    # Generate the meshgrid
    gx, gy, gz = tf.meshgrid(x_vals, y_vals, z_vals, indexing='ij')  # [21, 21, 5] each

    # Flatten the grid and stack to shape [N, 3]
    grid = tf.stack([
        tf.reshape(gx, [-1]),
        tf.reshape(gy, [-1]),
        tf.reshape(gz, [-1])
    ], axis=-1)  # [N, 3]

    # Repeat grid for each item in the batch
    grid_batched = tf.tile(tf.expand_dims(grid, axis=0), [batch_size, 1, 1])  # [B, N, 3]

    return grid_batched

def generate_events(centroids, event_gen, DEBUG=True, energy_range=(10., 150.)):
    '''
    Generate events based on centroids and event generator function.
    Args:
    - centroids: Tensor of shape [B, N, 3] with centroid coordinates
    **
    - event_gen: function to generate events
    - DEBUG: boolean flag for debug mode
    - energy_range: tuple of min and max energy values for event generation
    Returns:
    - inputs: Tensor of shape [B, N, 5] with input features
    - targets: Tensor of shape [B, N] with target signal fractions
    - E0: Tensor of shape [B, N] with initial energy values
    '''
    E0 = tf.random.uniform((centroids.shape[0],), minval=energy_range[0], maxval=energy_range[1])
    deposits, volumes, energy = event_gen(centroids, E0)
    signal_frac = tf.divide(energy, deposits+1e-7)
    
    inputs = tf.concat([
        centroids, 
        volumes[..., tf.newaxis], 
        deposits,
        E0*tf.ones_like(deposits)
        ], axis=-1)
    targets = signal_frac
    return inputs, targets, E0

def get_feature_list_single(inputs, signal_fraction):
    """
    Build model input list for a single event in differentiable form.

    Args:
        inputs: tf.Tensor of shape (N, 6) — [x, y, z, dE, volume, E0]
        signal_fraction: tf.Tensor of shape (N,) — signal weights (0 to 1)

    Returns:
        List[tf.Tensor or tf.RaggedTensor]: input format expected by the model
    """
    x = inputs[:, 0]
    y = inputs[:, 1]
    z = inputs[:, 2]
    volume = inputs[:, 3]
    dE = inputs[:, 4]
    E0 = inputs[0, 5]  # constant per event

    N = tf.shape(inputs)[0]
    row_splits = tf.convert_to_tensor([0, N], dtype=tf.int64)

    # Construct feature tensor matching training format
    features = tf.stack([
        dE,
        tf.zeros_like(dE),
        volume,
        x,
        y,
        z,
        tf.zeros_like(dE),
        tf.zeros_like(dE)
    ], axis=-1)

    # Targets (only used during loss eval)
    t_idx = tf.where(signal_fraction > 0.5, 1.0, 0.0)
    t_energy = tf.fill([N], E0)
    t_pos = tf.zeros((N, 3), dtype=tf.float32)
    t_time = tf.zeros((N, 1), dtype=tf.float32)
    t_pid = tf.zeros((N, 6), dtype=tf.float32)
    t_spectator = tf.zeros([N, 1], dtype=tf.float32)
    t_fully_contained = tf.ones([N, 1], dtype=tf.float32)
    t_rec_energy = tf.expand_dims(signal_fraction * dE, axis=-1)
    t_is_unique = tf.zeros([N, 1], dtype=tf.float32)

    t_sig_fraction = tf.expand_dims(signal_fraction, axis=-1)

    return [
        features,               row_splits,
        t_idx,                  row_splits,
        t_energy,               row_splits,
        t_pos,                  row_splits,
        t_time,                 row_splits,
        t_pid,                  row_splits,
        t_spectator,            row_splits,
        t_fully_contained,      row_splits,
        t_rec_energy,           row_splits,
        t_is_unique,            row_splits,
        t_sig_fraction,         row_splits
    ]

