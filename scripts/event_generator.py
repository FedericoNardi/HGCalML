import ROOT
import numpy as np
import os
import tensorflow as tf
from array import array
from genModules.Event import EventGenerator_v2 as EventGenerator

def generate_debug_grid_centroids(batch_size):
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

def generate_event(min_centroids=300., max_centroids=1000., energy_range=(10., 150.), DEBUG=True):
    B = 1
    # n_centroids = tf.random.uniform([B], min_centroids, max_centroids + 1)
    event_gen = EventGenerator()
    E0 = tf.random.uniform((1,), minval=energy_range[0], maxval=energy_range[1])
    centroids = generate_debug_grid_centroids(B) # tf.random.uniform([n_centroids, 3], minval=(-300.,-300., 0.) , maxval=(300., 300., 600.))
    deposits, volumes, energy = event_gen(centroids, E0)
    signal_frac = tf.where(deposits > 0, energy / deposits, 0.)
    print(centroids.shape)
    print(deposits.shape)
    if(DEBUG):
        plot_event(centroids[0].numpy(), deposits[0].numpy()+1e-6, save_path='img/debug/generated_vis_full')
        plot_event(centroids[0].numpy(), energy[0].numpy()+1e-6, save_path='img/debug/generated_vis_signal')
    
    inputs = tf.concat([
        centroids, 
        volumes[..., tf.newaxis], 
        deposits,
        E0*tf.ones_like(deposits)
        ], axis=-1)
    targets = signal_frac[..., tf.newaxis]
    return inputs.numpy().astype(np.float32), targets.numpy().astype(np.float32)

def write_root_file(filename, n_events=100, min_centroids=300, max_centroids=1000, max_centroids_limit=2500):
    f = ROOT.TFile(filename, "RECREATE", "", 0)
    tree = ROOT.TTree("Events", "Calorimeter Events")

    max_size = max_centroids_limit
    x = array('f', [0.0]*max_size)
    y = array('f', [0.0]*max_size)
    z = array('f', [0.0]*max_size)
    dE = array('f', [0.0]*max_size)
    vol = array('f', [0.0]*max_size)
    E0 = array('f', [0.0]*max_size)
    signal_fraction = array('f', [0.0]*max_size)
    n = array('i', [0])

    tree.Branch("n", n, "n/I")
    tree.Branch("x", x, "x[n]/F")
    tree.Branch("y", y, "y[n]/F")
    tree.Branch("z", z, "z[n]/F")
    tree.Branch("dE", dE, "dE[n]/F")
    tree.Branch("volume", vol, "volume[n]/F")
    tree.Branch("E0", E0, "E0[n]/F")
    tree.Branch("signal_fraction", signal_fraction, "signal_fraction[n]/F")

    for i in range(n_events):
        inputs, fracs = generate_event(min_centroids, max_centroids, DEBUG=False)

        n_centroids = inputs.shape[0]
        if n_centroids > max_centroids_limit:
            print(f"[WARN] Skipping event with {n_centroids} centroids (limit is {max_centroids_limit})")
            continue

        n[0] = n_centroids
        _x = inputs[:,0]
        _y = inputs[:,1]
        _z = inputs[:,2]
        _dE = inputs[:,3]
        _vol = inputs[:,4]
        _E0 = inputs[:,5]
        _signal_fraction = fracs[:,0]

        x[:n_centroids] = array('f', _x.tolist())
        y[:n_centroids] = array('f', _y.tolist())
        z[:n_centroids] = array('f', _z.tolist())
        dE[:n_centroids] = array('f', _dE.tolist())
        vol[:n_centroids] = array('f', _vol.tolist())
        E0[:n_centroids] = array('f', _E0.tolist())
        signal_fraction[:n_centroids] = array('f', _signal_fraction)

        tree.Fill()
        print(f"[{i+1}/{n_events}] Event with {n_centroids} centroids")

    tree.Write()
    f.Close()

def generate_multiple_root_files(output_dir, n_files=10, events_per_file=500, min_centroids=300, max_centroids=1000):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(n_files):
        filename = os.path.join(output_dir, f"events_{i:03d}.root")
        write_root_file(filename, events_per_file, min_centroids, max_centroids)

# --- DEBUGGING STUFF ---
def plot_event(centroids, bib_density, event_id=1, save_path="img/debug/generated_vis_v2"):
    """
    Plot projections of BIB energy density across centroids.

    Parameters:
    - centroids: np.ndarray of shape (N, 3)
    - bib_density: np.ndarray of shape (N,) or (N, 1)
    - event_id: int
    - save_path: output file prefix
    """
    import matplotlib.pyplot as plt
    import os
    os.makedirs("img/debug", exist_ok=True)

    x, y, z = centroids[:, 0], centroids[:, 1], centroids[:, 2]
    bib_density = bib_density.flatten()
    log_rho = np.log10(bib_density + 1e-9)

    fig = plt.figure(figsize=(24, 6))

    # 3D scatter
    ax = fig.add_subplot(141, projection='3d')
    sc = ax.scatter(x, y, z, c=log_rho, cmap='cividis', s=8, alpha=0.8)
    fig.colorbar(sc, ax=ax, label='log10(BIB density)')
    ax.set_title("BIB Density (3D)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # X-Z projection
    ax2 = fig.add_subplot(142)
    h_xz = ax2.hist2d(x, z, bins=(20,5), weights=bib_density, cmap='cividis',
                      norm=plt.matplotlib.colors.LogNorm())
    fig.colorbar(h_xz[3], ax=ax2, label='BIB density [X-Z]')
    ax2.set_title("X-Z Projection")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Z")

    # Y-Z projection
    ax3 = fig.add_subplot(143)
    h_yz = ax3.hist2d(y, z, bins=(20,5), weights=bib_density, cmap='cividis',
                      norm=plt.matplotlib.colors.LogNorm())
    fig.colorbar(h_yz[3], ax=ax3, label='BIB density [Y-Z]')
    ax3.set_title("Y-Z Projection")
    ax3.set_xlabel("Y")
    ax3.set_ylabel("Z")

    # X-Y projection
    ax4 = fig.add_subplot(144)
    h_xy = ax4.hist2d(x, y, bins=(20,20), weights=bib_density, cmap='cividis',
                      norm=plt.matplotlib.colors.LogNorm())
    fig.colorbar(h_xy[3], ax=ax4, label='BIB density [X-Y]')
    ax4.set_title("X-Y Projection")
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")

    plt.suptitle(f"BIB Energy Density (Event {event_id})", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{save_path}_{event_id:03d}.jpg", dpi=250)
    plt.close()

def plot_generated_event(generator_fn, save_path="img/debug/generated_vis.jpg"):
    """
    generator_fn: function like generate_event(...)
    save_path: where to save the generated plot
    """
    inputs, _ = generator_fn()

    x = inputs[:, 0]
    y = inputs[:, 1]
    z = inputs[:, 2]
    dE = inputs[:, 3]
    vol = inputs[:, 4]

    plot_event(x, y, z, vol, dE, title="Generated Event (direct)", save_path=save_path)

# --- DONE ---

if __name__ == "__main__":
    generate_event(min_centroids=300, max_centroids=1000, DEBUG=True)
    # generate_multiple_root_files("/media/disk/pyroot_tuples", n_files=20, events_per_file=500)

