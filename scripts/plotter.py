import ROOT
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from mpl_toolkits.mplot3d import Axes3D


def plot_event(x, y, z, volume, energy_deposit, title="Shower Event", save_path="img/generator_vis.jpg"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig = plt.figure(figsize=(18, 6))
    
    dE = np.asarray(energy_deposit)
    volume = np.asarray(volume)
    log_dE = np.log10(dE + 1e-6)
    norm_vol = volume / (np.max(volume) + 1e-6)

    # 1. 3D Scatter
    ax = fig.add_subplot(131, projection='3d')
    sc = ax.scatter(x, y, z, c=log_dE, cmap='viridis', s=norm_vol * 50, alpha=0.8)
    fig.colorbar(sc, ax=ax, label='log10(E deposit)')
    ax.set_title("3D Shower View")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # 2. X-Z hist2d
    ax2 = fig.add_subplot(132)
    h_xz = ax2.hist2d(x, z, bins=50, weights=dE, cmap='viridis', norm=plt.matplotlib.colors.LogNorm())
    fig.colorbar(h_xz[3], ax=ax2, label='E deposit [X-Z]')
    ax2.set_title("X-Z Projection (hist2d)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("z")

    # 3. Y-Z hist2d
    ax3 = fig.add_subplot(133)
    h_yz = ax3.hist2d(y, z, bins=50, weights=dE, cmap='viridis', norm=plt.matplotlib.colors.LogNorm())
    fig.colorbar(h_yz[3], ax=ax3, label='E deposit [Y-Z]')
    ax3.set_title("Y-Z Projection (hist2d)")
    ax3.set_xlabel("y")
    ax3.set_ylabel("z")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def load_and_plot_random_event(file_path):
    f = ROOT.TFile.Open(file_path)
    tree = f.Get("Events")
    n_entries = tree.GetEntries()

    idx = random.randint(0, n_entries - 1)
    tree.GetEntry(idx)

    n = tree.n
    if n == 0:
        print(f"Event {idx} has no centroids.")
        return

    x = np.frombuffer(tree.x, dtype=np.float32, count=n)
    y = np.frombuffer(tree.y, dtype=np.float32, count=n)
    z = np.frombuffer(tree.z, dtype=np.float32, count=n)
    vol = np.frombuffer(tree.volume, dtype=np.float32, count=n)
    dE = np.frombuffer(tree.dE, dtype=np.float32, count=n)

    plot_event(x, y, z, vol, dE, title=f"Event #{idx} from {file_path}")



load_and_plot_random_event("output_root_pyroot/events_000.root")