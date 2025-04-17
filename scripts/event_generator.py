import ROOT
import numpy as np
import os
import tensorflow as tf
from array import array
from genModules.Event import EventGenerator

def generate_debug_grid_centroids(bounds=(-200., 200.), bins_per_axis=8):
    """
    Generate a smaller, coarse 3D grid of centroids for debugging.

    Parameters:
    - bounds: (min, max) range in x, y, z
    - bins_per_axis: number of bins along each axis

    Returns:
    - Tensor of shape [N, 3] with centroid coordinates
    """
    min_b, max_b = bounds
    coords_1d = tf.linspace(min_b, max_b, bins_per_axis)
    gx, gy, gz = tf.meshgrid(coords_1d, coords_1d, coords_1d+200)
    grid = tf.stack([
        tf.reshape(gx, [-1]),
        tf.reshape(gy, [-1]),
        tf.reshape(gz, [-1])
    ], axis=-1)
    return grid

def generate_event(min_centroids=300, max_centroids=1000, energy_range=(10., 150.)):
    n_centroids = np.random.randint(min_centroids, max_centroids + 1)
    event_gen = EventGenerator()
    E0 = tf.random.uniform((1,), minval=energy_range[0], maxval=energy_range[1])
    centroids = generate_debug_grid_centroids() # tf.random.uniform([n_centroids, 3], minval=(-200.,-200., 0.) , maxval=(200., 200., 400.))
    deposits, volumes, energy = event_gen(centroids, E0)
    signal_frac = tf.where(deposits > 0, energy / deposits, 0.)
    print(centroids[:,0].shape)
    print(volumes.shape)
    print(energy.shape)
    plot_event(centroids.numpy(), deposits.numpy(), save_path='img/debug/generated_vis_full')
    plot_event(centroids.numpy(), energy.numpy(), save_path='img/debug/generated_vis_signal')
    
    inputs = tf.concat([
        centroids, 
        volumes[..., tf.newaxis], 
        deposits,
        E0*tf.ones_like(deposits)
        ], axis=-1)
    targets = signal_frac[..., tf.newaxis]
    return inputs.numpy().astype(np.float32), targets.numpy().astype(np.float32)

def write_root_file(filename, n_events=100, min_centroids=300, max_centroids=1000, max_centroids_limit=2000):
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
        inputs, fracs = generate_event(min_centroids, max_centroids)

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

def generate_multiple_root_files(output_dir, n_files=5, events_per_file=100, min_centroids=300, max_centroids=1000):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(n_files):
        filename = os.path.join(output_dir, f"events_{i:03d}.root")
        write_root_file(filename, events_per_file, min_centroids, max_centroids)

# --- DEBUGGING STUFF ---
def plot_event(centroids, bib_density, event_id=0, save_path="img/debug/generated_vis"):
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
    h_xz = ax2.hist2d(x, z, bins=8, weights=bib_density, cmap='cividis',
                      norm=plt.matplotlib.colors.LogNorm())
    fig.colorbar(h_xz[3], ax=ax2, label='BIB density [X-Z]')
    ax2.set_title("X-Z Projection")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Z")

    # Y-Z projection
    ax3 = fig.add_subplot(143)
    h_yz = ax3.hist2d(y, z, bins=8, weights=bib_density, cmap='cividis',
                      norm=plt.matplotlib.colors.LogNorm())
    fig.colorbar(h_yz[3], ax=ax3, label='BIB density [Y-Z]')
    ax3.set_title("Y-Z Projection")
    ax3.set_xlabel("Y")
    ax3.set_ylabel("Z")

    # X-Y projection
    ax4 = fig.add_subplot(144)
    h_xy = ax4.hist2d(x, y, bins=8, weights=bib_density, cmap='cividis',
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
    generate_event(min_centroids=300, max_centroids=1000)
    # generate_multiple_root_files("output_root_pyroot", n_files=5, events_per_file=10)

