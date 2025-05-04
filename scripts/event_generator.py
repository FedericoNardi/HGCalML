import os
import tensorflow as tf
import numpy as np
import ROOT
from array import array
from genModules import EventGenerator_v2 as EventGenerator

# --- Event generation ---
def generate_debug_grid_centroids(batch_size, smear_stddev=1.0):
    z_vals = tf.linspace(0., 200., 5)
    x_vals = tf.linspace(-400., 400., 21)
    y_vals = tf.linspace(-400., 400., 21)

    gx, gy, gz = tf.meshgrid(x_vals, y_vals, z_vals, indexing='ij')

    grid = tf.stack([
        tf.reshape(gx, [-1]),
        tf.reshape(gy, [-1]),
        tf.reshape(gz, [-1])
    ], axis=-1)

    grid_batched = tf.tile(tf.expand_dims(grid, axis=0), [batch_size, 1, 1])

    if smear_stddev > 0.0:
        noise = tf.random.normal(shape=grid_batched.shape, mean=0.0, stddev=smear_stddev)
        grid_batched += noise

    return grid_batched


def generate_event(min_centroids=300., max_centroids=1000., energy_range=(10., 150.), DEBUG=True, batch_size=5):
    event_gen = EventGenerator()
    E0 = tf.random.uniform((batch_size,), minval=energy_range[0], maxval=energy_range[1])
    centroids = generate_debug_grid_centroids(batch_size)
    deposits, volumes, energy = event_gen(centroids, E0)
    signal_frac = tf.where(deposits > 0, energy / deposits, 0.)

    if DEBUG:
        plot_event(centroids[0].numpy(), deposits[0].numpy()+1e-6, save_path='img/debug/generated_vis_full')
        plot_event(centroids[0].numpy(), energy[0].numpy()+1e-6, save_path='img/debug/generated_vis_signal')

    inputs = tf.concat([
        centroids, 
        volumes[..., tf.newaxis], 
        deposits,
        E0[:, tf.newaxis, tf.newaxis]*tf.ones_like(deposits)
    ], axis=-1)

    targets = signal_frac  # already (B, N)

    return inputs.numpy().astype(np.float32), targets.numpy().astype(np.float32)


# --- ROOT writing ---
def write_root_file(filename, n_events=100, min_centroids=300, max_centroids=1000, max_centroids_limit=2500, batch_size=10):
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

    events_written = 0

    while events_written < n_events:
        inputs_batch, fracs_batch = generate_event(min_centroids, max_centroids, DEBUG=False, batch_size=batch_size)
        current_batch_size = inputs_batch.shape[0]

        for b in range(current_batch_size):
            if events_written >= n_events:
                break

            inputs = inputs_batch[b]
            fracs = fracs_batch[b]

            n_centroids = inputs.shape[0]
            if n_centroids > max_centroids_limit:
                print(f"[WARN] Skipping event with {n_centroids} centroids (limit is {max_centroids_limit})")
                continue

            n[0] = n_centroids

            x[:n_centroids] = array('f', inputs[:,0].tolist())
            y[:n_centroids] = array('f', inputs[:,1].tolist())
            z[:n_centroids] = array('f', inputs[:,2].tolist())
            dE[:n_centroids] = array('f', inputs[:,3].tolist())
            vol[:n_centroids] = array('f', inputs[:,4].tolist())
            E0[:n_centroids] = array('f', inputs[:,5].tolist())
            signal_fraction[:n_centroids] = array('f', np.reshape(fracs, (-1,)).tolist())

            tree.Fill()
            events_written += 1

            print(f"[{events_written}/{n_events}] Event with {n_centroids} centroids")

    tree.Write()
    f.Close()


def generate_multiple_root_files(output_dir, n_files=10, events_per_file=500, min_centroids=300, max_centroids=1000, batch_size=5):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(n_files):
        filename = os.path.join(output_dir, f"events_{i:03d}.root")
        write_root_file(filename, events_per_file, min_centroids, max_centroids, batch_size=5)


# --- Plotting ---
def plot_event(centroids, bib_density, event_id=1, save_path="img/debug/generated_vis_v2"):
    import matplotlib.pyplot as plt
    os.makedirs("img/debug", exist_ok=True)

    x, y, z = centroids[:, 0], centroids[:, 1], centroids[:, 2]
    bib_density = bib_density.flatten()
    log_rho = np.log10(bib_density + 1e-9)

    fig = plt.figure(figsize=(24, 6))

    ax = fig.add_subplot(141, projection='3d')
    sc = ax.scatter(x, y, z, c=log_rho, cmap='cividis', s=8, alpha=0.8)
    fig.colorbar(sc, ax=ax, label='log10(BIB density)')
    ax.set_title("BIB Density (3D)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax2 = fig.add_subplot(142)
    h_xz = ax2.hist2d(x, z, bins=(20,5), weights=bib_density, cmap='cividis',
                      norm=plt.matplotlib.colors.LogNorm())
    fig.colorbar(h_xz[3], ax=ax2, label='BIB density [X-Z]')
    ax2.set_title("X-Z Projection")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Z")

    ax3 = fig.add_subplot(143)
    h_yz = ax3.hist2d(y, z, bins=(20,5), weights=bib_density, cmap='cividis',
                      norm=plt.matplotlib.colors.LogNorm())
    fig.colorbar(h_yz[3], ax=ax3, label='BIB density [Y-Z]')
    ax3.set_title("Y-Z Projection")
    ax3.set_xlabel("Y")
    ax3.set_ylabel("Z")

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


# --- Entry point ---
if __name__ == "__main__":
    generate_multiple_root_files("/media/disk/pyroot_tuples_smear", n_files=50, events_per_file=500, batch_size=10)
